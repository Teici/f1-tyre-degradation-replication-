import argparse
import pathlib
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from src.data import to_supervised_tabular
from src.utils import set_seed, ensure_dir, save_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    df = pd.read_parquet(args.data) if args.data.endswith(".parquet") else pd.read_csv(args.data)
    tab = to_supervised_tabular(df)

    target = "y_degradation"
    group = tab["stint_id"].to_numpy()

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train_idx, test_idx = next(gss.split(tab, groups=group))
    train = tab.iloc[train_idx].copy()
    test  = tab.iloc[test_idx].copy()

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed+1)
    tr_idx, va_idx = next(gss2.split(train, groups=train["stint_id"].to_numpy()))
    tr = train.iloc[tr_idx].copy()
    va = train.iloc[va_idx].copy()

    cat_cols = ["compound"]
    drop_cols = {"stint_id", "lap", target, "lap_time", "lap_time_ref", "compound"}
    num_cols = [c for c in tab.columns if c not in drop_cols]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ])

    models = {
        "linear": LinearRegression(),
        "rf": RandomForestRegressor(
            n_estimators=300,
            random_state=args.seed,
            n_jobs=-1,
            min_samples_leaf=2,
        )
    }

    out_dir = pathlib.Path(args.out_dir)
    ensure_dir(str(out_dir))

    meta = {"target": target, "cat_cols": cat_cols, "num_cols": num_cols, "seed": args.seed}
    save_json(meta, str(out_dir / "meta.json"))

    for name, mdl in models.items():
        pipe = Pipeline([("pre", pre), ("model", mdl)])
        pipe.fit(tr[cat_cols + num_cols], tr[target])
        joblib.dump(pipe, out_dir / f"{name}.joblib")
        print(f"Saved {name} model")

    np.save(out_dir / "test_stint_ids.npy", test["stint_id"].unique())
    np.save(out_dir / "val_stint_ids.npy", va["stint_id"].unique())
    print("Saved val/test split stint ids.")

if __name__ == "__main__":
    main()
