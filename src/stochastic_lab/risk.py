"""Simulation-based risk metrics."""

from __future__ import annotations

import numpy as np


def value_at_risk(losses: np.ndarray, confidence: float = 0.99) -> float:
    """Empirical loss quantile."""
    x = np.asarray(losses, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("losses must be a non-empty one-dimensional array")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    return float(np.quantile(x, confidence))


def expected_shortfall(losses: np.ndarray, confidence: float = 0.99) -> float:
    """Empirical expected shortfall beyond VaR."""
    x = np.asarray(losses, dtype=float)
    var = value_at_risk(x, confidence)
    tail = x[x >= var]
    return float(tail.mean())


def drawdown(path: np.ndarray) -> np.ndarray:
    """Fractional drawdown series relative to the running maximum."""
    x = np.asarray(path, dtype=float)
    if x.ndim != 1 or x.size == 0 or np.any(x <= 0):
        raise ValueError("path must be a non-empty positive one-dimensional array")
    running_max = np.maximum.accumulate(x)
    return 1.0 - x / running_max


def maximum_drawdown(path: np.ndarray) -> float:
    """Maximum fractional drawdown."""
    return float(drawdown(path).max())
