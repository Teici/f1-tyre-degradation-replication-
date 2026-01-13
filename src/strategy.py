import numpy as np


def simulate_stint_strategy(
    lap_times: np.ndarray,
    degradation: np.ndarray,
    predict_fn,
    horizon: int = 5,
    threshold: float = 1.2,
):
    """
    Simulate a simple pit stop strategy for a single stint.

    Parameters
    ----------
    lap_times : np.ndarray
        True lap times for the stint.
    degradation : np.ndarray
        True degradation signal (lap_time - reference_lap).
    predict_fn : callable
        Function returning predicted degradation array.
    horizon : int
        How many laps ahead the strategy looks.
    threshold : float
        Degradation threshold that triggers a pit stop.

    Returns
    -------
    pit_lap : int
        Lap at which the pit stop is triggered.
    total_time : float
        Accumulated lap time until pit decision.
    """
    total_time = 0.0
    pit_lap = len(lap_times) - 1  # default: pit at end

    for lap in range(len(lap_times)):
        total_time += lap_times[lap]

        if lap + horizon >= len(degradation):
            continue

        future_pred = predict_fn(lap)[lap + horizon]

        if future_pred >= threshold:
            pit_lap = lap
            break

    return pit_lap, total_time
