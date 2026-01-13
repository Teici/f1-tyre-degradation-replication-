

# AI-Based Tyre Degradation Prediction and Pit Stop Strategy Evaluation  
## Partial Replication + Extension Study (Formula 1)

This repository contains the implementation and experiments for a **partial replication + extension study** aligned with my previous reports on:
- **Tyre degradation prediction** using ML/DL (notably LSTM time-series models), and  
- **Pit stop decision-making** as part of an integrated workflow (prediction → strategy).

Because real Formula 1 telemetry is proprietary, experiments are conducted on a **simulation-generated stint dataset**, consistent with common practice described in the literature.

---

## 1) Study Type

### Replication component (Tyre degradation prediction)
We replicate common experimental patterns reported in the literature:
- Formulate tyre degradation prediction as **regression** over lap sequences.
- Compare **classical ML baselines** with a **time-series LSTM** model.
- Evaluate using standard metrics: **MAE** and **RMSE** (MAPE also reported).

Models:
- Linear Regression (baseline)
- Random Forest (baseline)
- LSTM regressor (sequence model)

### What is different from prior work
- **Dataset:** simulated/synthetic stints instead of proprietary telemetry.
- **Tooling:** modern Python / PyTorch / scikit-learn versions and a reproducible training pipeline.

### Extension component (Strategy impact)
We extend the replicated experiments by adding a controlled strategy evaluation:
- A **rule-based pit stop policy** that pits when *predicted* future degradation crosses a threshold.
- Compare strategy outcomes using:
  - Oracle degradation (ground truth)
  - Linear/RF predictions
  - LSTM predictions

This quantifies how prediction errors propagate into pit timing and total time, supporting the integrated pipeline perspective described in the literature.

---

## 2) Repository Structure

```text
src/                    Core modules (data simulation, models, strategy, utils)
experiments/            Experiment scripts (CLI entry points)
data/processed/         Generated datasets
artifacts/              Trained models + metadata
results/                Metrics, plots, CSV outputs
docker/                 Optional Dockerfile
````

---

## 3) Requirements

* Python 3.10+ (recommended 3.11)
* CPU-only works
* Optional GPU acceleration (PyTorch will use CUDA automatically if available)

---

## 4) Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv

# Windows:
# .venv\Scripts\activate

# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

---

## 5) Reproducibility Notes

* Random seeds are controlled using `--seed`.
* Splits are performed **by stint_id** (not by lap) to avoid leakage across the same stint.
* Model metadata is saved to:

  * `artifacts/baselines/meta.json`
  * `artifacts/lstm/meta.json`

---

## 6) Experiment 1 — Dataset Generation

This project uses a **simulation-generated stint dataset** because real F1 telemetry is not publicly available.

Generate the dataset:

```bash
python -m experiments.generate_data \
  --out data/processed/stints.parquet \
  --n_stints 8000 \
  --seed 42
```

Output:

* `data/processed/stints.parquet`

---

## 7) Experiment 2 — Tyre Degradation Prediction (Replication)

### 7.1 Train baselines

```bash
python -m experiments.train_baselines \
  --data data/processed/stints.parquet \
  --out_dir artifacts/baselines \
  --seed 42
```

Outputs:

* `artifacts/baselines/linear.joblib`
* `artifacts/baselines/rf.joblib`
* `artifacts/baselines/meta.json`
* `artifacts/baselines/val_stint_ids.npy`
* `artifacts/baselines/test_stint_ids.npy`

### 7.2 Train LSTM

```bash
python -m experiments.train_lstm \
  --data data/processed/stints.parquet \
  --out_dir artifacts/lstm \
  --baseline_split_dir artifacts/baselines \
  --epochs 30 \
  --seed 42
```

Outputs:

* `artifacts/lstm/best.pt`
* `artifacts/lstm/meta.json`
* `artifacts/lstm/train_summary.json`

### 7.3 Evaluate prediction metrics

```bash
python -m experiments.evaluate \
  --data data/processed/stints.parquet \
  --baseline_dir artifacts/baselines \
  --lstm_dir artifacts/lstm \
  --out_dir results
```

Outputs:

* `results/metrics.json`
* `results/metrics.csv`
* `results/plots/lstm_scatter.png`

### Example results (from `results/metrics.json`)

| Model  | MAE   | RMSE  | MAPE  |
| ------ | ----- | ----- | ----- |
| Linear | 0.227 | 0.286 | 42.78 |
| RF     | 0.229 | 0.287 | 40.67 |
| LSTM   | 0.291 | 0.365 | 23.01 |

Scatter plot:

* `results/plots/lstm_scatter.png`

---

## 8) Experiment 3 — Pit Stop Strategy Impact (Extension)

### Strategy definition

A simple rule-based policy is used:

> **Pit when predicted degradation (H laps ahead) exceeds a threshold.**

This isolates the effect of degradation prediction quality on pit timing and total time.

### Run the strategy evaluation

```bash
python -m experiments.evaluate_strategy \
  --data data/processed/stints.parquet \
  --baseline_dir artifacts/baselines \
  --lstm_dir artifacts/lstm \
  --out results/strategy_metrics.json
```

Outputs:

* `results/strategy_metrics.json` (summary statistics)
* `results/strategy_metrics.csv` (per-stint outcomes)

### Example strategy summary (from `results/strategy_metrics.json`)

* Evaluated stints: 7380
* Parameters:

  * horizon = 5
  * threshold = 1.2

Mean pit-lap deviation vs oracle:

* Linear: -1.16 laps
* RF:     -1.43 laps
* LSTM:   -3.30 laps

Mean time loss vs oracle (negative = faster than oracle under this policy/simulator):

* Linear: -108.26
* RF:     -134.08
* LSTM:   -309.25

**Interpretation:** negative time loss indicates earlier pit decisions that reduce stint time under the simulated conditions. This demonstrates that prediction outputs can materially shift pit decisions and performance.

---

## 9) Troubleshooting

### “The code is stuck”

Some steps can be slow depending on CPU resources, especially:

* Random Forest training
* LSTM training (many epochs)

Try a smaller run:

```bash
python -m experiments.generate_data --out data/processed/stints_small.parquet --n_stints 1000 --seed 42
python -m experiments.train_baselines --data data/processed/stints_small.parquet --out_dir artifacts/baselines_small --seed 42
python -m experiments.train_lstm --data data/processed/stints_small.parquet --out_dir artifacts/lstm_small --baseline_split_dir artifacts/baselines_small --epochs 5 --seed 42
python -m experiments.evaluate --data data/processed/stints_small.parquet --baseline_dir artifacts/baselines_small --lstm_dir artifacts/lstm_small --out_dir results_small
```

---

## 10) Optional: Docker

Build:

```bash
docker build -t f1-tyre-replication -f docker/Dockerfile .
```

Run (example):

```bash
docker run --rm -it f1-tyre-replication python -m experiments.generate_data --help
```

---

