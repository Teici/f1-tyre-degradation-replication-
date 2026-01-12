import argparse
import pathlib
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.data import to_supervised_tabular
from src.lstm_model import LSTMRegressor
from src.sequence import StintSequenceDataset
from src.utils import ensure_dir, load_json, save_json, mae, rmse, mape

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--baseline_dir", type=str, required=True)
    ap.add_argument("--lstm_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.data) if args.data.endswith(".parquet") else pd.read_csv(args.data)

    baseline_dir = pathlib.Path(args.baseline_dir)
    lstm_dir = pathlib.Path(args.lstm_dir)
    out_dir = pathlib.Path(args.out_dir)
    ensure_dir(str(out_dir))
    ensure_dir(str(out_dir / "plots"))

    # --- Baselines ---
    tab = to_supervised_tabular(df)
    meta_b = load_json(baseline_dir / "meta.json")
    cat_cols = meta_b["cat_cols"]
    num_cols = meta_b["num_cols"]
    test_ids = np.load(baseline_dir / "test_stint_ids.npy")
    test_tab = tab[tab["stint_id"].isin(test_ids)]

    X = test_tab[cat_cols + num_cols]
    y = test_tab["y_degradation"].to_numpy()

    baseline_metrics = {}
    for name in ["linear", "rf"]:
        model = joblib.load(baseline_dir / f"{name}.joblib")
        pred = model.predict(X)
        baseline_metrics[name] = {"MAE": mae(y,pred), "RMSE": rmse(y,pred), "MAPE": mape(y,pred)}

    # --- LSTM ---
    meta_l = load_json(lstm_dir / "meta.json")
    feature_cols = meta_l["feature_cols"]
    target_col = meta_l["target_col"]
    seq_len = int(meta_l["seq_len"])

    df = df.sort_values(["stint_id","lap"]).copy()
    df["compound_id"] = df["compound"].map({"SOFT":0.0,"MEDIUM":1.0,"HARD":2.0}).astype(float)

    mu = pd.Series(meta_l["normalization"]["mu"])
    sd = pd.Series(meta_l["normalization"]["sd"]).replace(0,1.0)
    norm_cols = [c for c in feature_cols if c not in ("lap","compound_id")]
    df[norm_cols] = (df[norm_cols] - mu[norm_cols]) / sd[norm_cols]

    df_test = df[df["stint_id"].isin(test_ids)].copy()
    ds = StintSequenceDataset(df_test, feature_cols, target_col, seq_len)
    loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMRegressor(n_features=len(feature_cols))
    model.load_state_dict(torch.load(lstm_dir / "best.pt", map_location=device))
    model.to(device)
    model.eval()

    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            ps.append(model(xb).cpu().numpy().reshape(-1))
            ys.append(yb.numpy().reshape(-1))
    y_l = np.concatenate(ys)
    p_l = np.concatenate(ps)

    lstm_metrics = {"MAE": mae(y_l,p_l), "RMSE": rmse(y_l,p_l), "MAPE": mape(y_l,p_l)}

    metrics = {"baselines": baseline_metrics, "lstm": lstm_metrics}
    save_json(metrics, str(out_dir / "metrics.json"))

    # plot
    n = min(2500, len(y_l))
    plt.figure()
    plt.scatter(y_l[:n], p_l[:n], s=6)
    plt.xlabel("True degradation (s)")
    plt.ylabel("Predicted degradation (s)")
    plt.tight_layout()
    plt.savefig(out_dir / "plots" / "lstm_scatter.png", dpi=200)
    plt.close()

    print("Saved:", out_dir / "metrics.json")

if __name__ == "__main__":
    main()
