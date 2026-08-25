"""Generic calibration-loss and parameter-diagnostic helpers."""

from __future__ import annotations

import numpy as np


def weighted_rmse(observed: np.ndarray, modelled: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Compute weighted root-mean-square calibration error."""
    observed = np.asarray(observed, dtype=float)
    modelled = np.asarray(modelled, dtype=float)
    if observed.shape != modelled.shape or observed.size == 0:
        raise ValueError("observed and modelled must have the same non-empty shape")
    residuals = modelled - observed
    if weights is None:
        return float(np.sqrt(np.mean(residuals**2)))
    weights = np.asarray(weights, dtype=float)
    if weights.shape != observed.shape or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weights must be non-negative, non-zero and match observations")
    weights = weights / weights.sum()
    return float(np.sqrt(np.sum(weights * residuals**2)))


def relative_rmse(observed: np.ndarray, modelled: np.ndarray, floor: float = 1e-12) -> float:
    """Compute RMSE of pointwise relative errors."""
    observed = np.asarray(observed, dtype=float)
    modelled = np.asarray(modelled, dtype=float)
    if observed.shape != modelled.shape or observed.size == 0:
        raise ValueError("observed and modelled must have the same non-empty shape")
    scale = np.maximum(np.abs(observed), floor)
    return float(np.sqrt(np.mean(((modelled - observed) / scale) ** 2)))


def parameter_bounds_check(parameters: dict[str, float], bounds: dict[str, tuple[float, float]]) -> dict[str, bool]:
    """Check named parameters against inclusive lower and upper bounds."""
    result: dict[str, bool] = {}
    for name, value in parameters.items():
        if name not in bounds:
            continue
        lower, upper = bounds[name]
        result[name] = bool(lower <= float(value) <= upper)
    return result
