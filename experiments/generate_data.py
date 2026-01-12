import argparse
import pathlib
from src.data import SimConfig, simulate_stints
from src.utils import set_seed, ensure_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--n_stints", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_laps", type=int, default=45)
    ap.add_argument("--min_laps", type=int, default=10)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = SimConfig(max_laps=args.max_laps, min_laps=args.min_laps, seed=args.seed)
    df = simulate_stints(args.n_stints, cfg)

    out_path = pathlib.Path(args.out)
    ensure_dir(str(out_path.parent))
    if out_path.suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}  shape={df.shape}")

if __name__ == "__main__":
    main()
