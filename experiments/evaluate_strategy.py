import argparse
import pathlib
import numpy as np
import pandas as pd
import joblib
import torch

from src.data import to_supervised_tabular
from src.strategy import simulate_stint_strategy
from src.lstm_model import LSTMRegressor
from src.utils import load_json, save_json


def build_lstm_sequences(df: pd.DataFrame, feature_cols: list, seq_len: int) -> torch.Tensor:
    """
    Build sliding-window sequences for LSTM inference.
    """
    X = []
    for i in range(seq_len - 1, len(df)):
        X.append(df.iloc[i - seq_len + 1 : i + 1][feature_cols].to_numpy())
    return torch.tensor(np.array(X), dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--lstm_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=1.2)
    args = parser.parse_args()

    df = pd.read_parquet(args.data)

    baseline_dir = pathlib.Path(args.baseline_dir)
    lstm_dir = pathlib.Path(args.lstm_dir)
    out_path = pathlib.Path(args.out)

    meta_b = load_json(baseline_dir / "meta.json")
    cat_cols = meta_b["cat_cols"]
    num_cols = meta_b["num_cols"]

    linear = joblib.load(baseline_dir / "linear.joblib")
    rf = joblib.load(baseline_dir / "rf.joblib")

    meta_l = load_json(lstm_dir / "meta.json")
    feature_cols = meta_l["feature_cols"]
    seq_len = int(meta_l["seq_len"])

    mu = pd.Series(meta_l["normalization"]["mu"])
    sd = pd.Series(meta_l["normalization"]["sd"]).replace(0, 1.0)

    compound_map = {"SOFT": 0.0, "MEDIUM": 1.0, "HARD": 2.0}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lstm = LSTMRegressor(n_features=len(feature_cols))
    lstm.load_state_dict(torch.load(lstm_dir / "best.pt", map_location=device))
    lstm.to(device)
    lstm.eval()

    results = []

    for stint_id, g in df.groupby("stint_id"):
        g = g.sort_values("lap").reset_index(drop=True)

        lap_times_full = g["lap_time"].to_numpy()
        true_deg_full = g["y_degradation"].to_numpy()

        oracle_fn = lambda _: true_deg_full
        pit_oracle, time_oracle = simulate_stint_strategy(
            lap_times_full,
            true_deg_full,
            oracle_fn,
            horizon=args.horizon,
            threshold=args.threshold,
        )

        tab = to_supervised_tabular(g)
        if len(tab) < 10:
            continue

        X_tab = tab[cat_cols + num_cols]
        lap_times_tab = tab["lap_time"].to_numpy()
        true_deg_tab = tab["y_degradation"].to_numpy()

        lin_pred = linear.predict(X_tab)
        rf_pred = rf.predict(X_tab)

        pit_lin, time_lin = simulate_stint_strategy(
            lap_times_tab,
            true_deg_tab,
            lambda _: lin_pred,
            horizon=args.horizon,
            threshold=args.threshold,
        )

        pit_rf, time_rf = simulate_stint_strategy(
            lap_times_tab,
            true_deg_tab,
            lambda _: rf_pred,
            horizon=args.horizon,
            threshold=args.threshold,
        )

        g_lstm = g.copy()
        g_lstm["compound_id"] = g_lstm["compound"].map(compound_map).astype(float)

        norm_cols = [c for c in feature_cols if c not in ("lap", "compound_id")]
        g_lstm[norm_cols] = (g_lstm[norm_cols] - mu[norm_cols]) / sd[norm_cols]

        if len(g_lstm) < seq_len:
            continue

        X_seq = build_lstm_sequences(g_lstm, feature_cols, seq_len).to(device)

        with torch.no_grad():
            lstm_pred = lstm(X_seq).cpu().numpy().flatten()

        lap_times_lstm = g["lap_time"].to_numpy()[seq_len - 1 :]
        true_deg_lstm = g["y_degradation"].to_numpy()[seq_len - 1 :]

        pit_lstm, time_lstm = simulate_stint_strategy(
            lap_times_lstm,
            true_deg_lstm,
            lambda _: lstm_pred,
            horizon=args.horizon,
            threshold=args.threshold,
        )

        results.append({
            "stint_id": int(stint_id),
            "pit_oracle": pit_oracle,
            "pit_linear": pit_lin,
            "pit_rf": pit_rf,
            "pit_lstm": pit_lstm,
            "oracle_time": time_oracle,
            "linear_time": time_lin,
            "rf_time": time_rf,
            "lstm_time": time_lstm,
        })

    out_df = pd.DataFrame(results)

    out_df["linear_loss"] = out_df["linear_time"] - out_df["oracle_time"]
    out_df["rf_loss"] = out_df["rf_time"] - out_df["oracle_time"]
    out_df["lstm_loss"] = out_df["lstm_time"] - out_df["oracle_time"]

    summary = {
        "n_stints": int(len(out_df)),
        "params": {
            "horizon": args.horizon,
            "threshold": args.threshold,
        },
        "mean_time_loss": {
            "linear": float(out_df["linear_loss"].mean()),
            "rf": float(out_df["rf_loss"].mean()),
            "lstm": float(out_df["lstm_loss"].mean()),
        },
        "median_time_loss": {
            "linear": float(out_df["linear_loss"].median()),
            "rf": float(out_df["rf_loss"].median()),
            "lstm": float(out_df["lstm_loss"].median()),
        },
        "pit_lap_diff_mean": {
            "linear": float((out_df["pit_linear"] - out_df["pit_oracle"]).mean()),
            "rf": float((out_df["pit_rf"] - out_df["pit_oracle"]).mean()),
            "lstm": float((out_df["pit_lstm"] - out_df["pit_oracle"]).mean()),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(summary, str(out_path))
    out_df.to_csv(out_path.with_suffix(".csv"), index=False)

    print(f"Saved strategy summary: {out_path}")
    print(f"Saved per-stint CSV:     {out_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
