from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]

@dataclass
class SimConfig:
    max_laps: int = 45
    min_laps: int = 10
    seed: int = 42

def _compound_params(compound: str):
    if compound == "SOFT":
        return dict(init_grip=1.00, wear_rate=1.25)
    if compound == "MEDIUM":
        return dict(init_grip=0.98, wear_rate=1.00)
    return dict(init_grip=0.96, wear_rate=0.80)

def simulate_stints(n_stints: int, cfg: SimConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    rows = []

    for stint_id in range(n_stints):
        track_abrasion = rng.uniform(0.7, 1.4)
        ambient_temp = rng.normal(30.0, 6.0)
        compound = rng.choice(COMPOUNDS, p=[0.38, 0.42, 0.20])
        params = _compound_params(compound)

        stint_len = int(rng.integers(cfg.min_laps, cfg.max_laps + 1))
        fuel_start = rng.uniform(90, 110)
        fuel_burn = rng.uniform(1.6, 2.2)
        driver_aggr = rng.uniform(0.85, 1.15)

        cum_wear = 0.0
        thermal = 0.0
        base_lap = rng.normal(90.0, 2.0)

        for lap in range(stint_len):
            fuel = max(fuel_start - fuel_burn * lap, 5.0)
            fuel_effect = 0.035 * fuel

            tyre_temp = ambient_temp + 35 * driver_aggr + rng.normal(0, 2.0)
            thermal = 0.85 * thermal + max(0.0, (tyre_temp - 95.0)) * 0.05

            wear_increment = (
                0.020 * params["wear_rate"] * track_abrasion * driver_aggr
                + 0.010 * thermal
                + rng.normal(0.0, 0.002)
            )
            wear_increment = max(0.0, wear_increment)
            cum_wear += wear_increment

            grip = params["init_grip"] - 0.35 * (1 - np.exp(-3.0 * cum_wear))
            grip = max(0.55, grip)

            degr_penalty = (1.0 / grip - 1.0) * 10.0 + 0.7 * thermal
            lap_time = base_lap + fuel_effect + degr_penalty + rng.normal(0, 0.25)

            rows.append(dict(
                stint_id=stint_id,
                lap=lap,
                compound=compound,
                track_abrasion=track_abrasion,
                ambient_temp=ambient_temp,
                fuel=fuel,
                driver_aggr=driver_aggr,
                tyre_temp=tyre_temp,
                thermal=thermal,
                cum_wear=cum_wear,
                lap_time=lap_time,
            ))

    df = pd.DataFrame(rows)
    df["lap_time_ref"] = df.groupby("stint_id")["lap_time"].transform("first")
    df["y_degradation"] = df["lap_time"] - df["lap_time_ref"]
    return df

def to_supervised_tabular(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["stint_id", "lap"]).copy()
    for k in [1, 2, 3]:
        df[f"lag{k}_tyre_temp"] = df.groupby("stint_id")["tyre_temp"].shift(k)
        df[f"lag{k}_degr"] = df.groupby("stint_id")["y_degradation"].shift(k)
    return df.dropna().reset_index(drop=True)
