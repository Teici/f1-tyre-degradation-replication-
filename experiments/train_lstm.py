import argparse
import pathlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from src.sequence import StintSequenceDataset
from src.lstm_model import LSTMRegressor
from src.utils import set_seed, ensure_dir, save_json, rmse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--baseline_split_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seq_len", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_parquet(args.data) if args.data.endswith(".parquet") else pd.read_csv(args.data)
    df = df.sort_values(["stint_id", "lap"]).copy()

    compound_map = {"SOFT": 0.0, "MEDIUM": 1.0, "HARD": 2.0}
    df["compound_id"] = df["compound"].map(compound_map).astype(float)

    feature_cols = ["lap","compound_id","track_abrasion","ambient_temp","fuel","driver_aggr","tyre_temp","thermal","cum_wear"]
    target_col = "y_degradation"

    split_dir = pathlib.Path(args.baseline_split_dir)
    val_ids = np.load(split_dir / "val_stint_ids.npy")
    test_ids = np.load(split_dir / "test_stint_ids.npy")
    all_ids = df["stint_id"].unique()
    train_ids = np.array([sid for sid in all_ids if sid not in set(val_ids) and sid not in set(test_ids)])

    df_train = df[df["stint_id"].isin(train_ids)].copy()
    df_val   = df[df["stint_id"].isin(val_ids)].copy()
    df_test  = df[df["stint_id"].isin(test_ids)].copy()

    norm_cols = [c for c in feature_cols if c not in ("lap","compound_id")]
    mu = df_train[norm_cols].mean()
    sd = df_train[norm_cols].std().replace(0, 1.0)

    for part in (df_train, df_val, df_test):
        part[norm_cols] = (part[norm_cols] - mu) / sd

    train_ds = StintSequenceDataset(df_train, feature_cols, target_col, args.seq_len)
    val_ds   = StintSequenceDataset(df_val, feature_cols, target_col, args.seq_len)
    test_ds  = StintSequenceDataset(df_test, feature_cols, target_col, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = LSTMRegressor(n_features=len(feature_cols))
    model.to(device)

    opt = AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    out_dir = pathlib.Path(args.out_dir)
    ensure_dir(str(out_dir))

    meta = {
        "feature_cols": feature_cols,
        "target_col": target_col,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "normalization": {"mu": mu.to_dict(), "sd": sd.to_dict()},
        "device": device
    }
    save_json(meta, str(out_dir / "meta.json"))

    best_val = float("inf")
    best_path = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for xb, yb in pbar:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                ps.append(model(xb).cpu().numpy().reshape(-1))
                ys.append(yb.numpy().reshape(-1))
        y = np.concatenate(ys); p = np.concatenate(ps)
        val_rmse = rmse(y, p)
        print(f"Epoch {epoch}: val_rmse={val_rmse:.4f}")

        if val_rmse < best_val:
            best_val = val_rmse
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            ps.append(model(xb).cpu().numpy().reshape(-1))
            ys.append(yb.numpy().reshape(-1))
    y = np.concatenate(ys); p = np.concatenate(ps)
    test_rmse = rmse(y, p)

    save_json({"best_val_rmse": best_val, "test_rmse": test_rmse}, str(out_dir / "train_summary.json"))
    print(f"Saved best model to {best_path} | test_rmse={test_rmse:.4f}")

if __name__ == "__main__":
    main()
