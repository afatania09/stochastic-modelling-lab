"""Heston backward pricing operator on supplied analytic derivatives.

The function deliberately does not use numerical differentiation or a neural
network: manufactured solutions can verify coefficient/sign conventions first.
"""
from __future__ import annotations

import numpy as np


def heston_backward_residual(
    *,
    spot: np.ndarray,
    variance: np.ndarray,
    rate: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    value: np.ndarray,
    dt: np.ndarray,
    ds: np.ndarray,
    dv: np.ndarray,
    dss: np.ndarray,
    dsv: np.ndarray,
    dvv: np.ndarray,
) -> np.ndarray:
    """Return V_t + L_Heston V - r V for a zero-dividend underlying."""
    if kappa <= 0 or theta <= 0 or xi <= 0 or not -1 <= rho <= 1:
        raise ValueError("invalid Heston parameters")
    s, v = np.broadcast_arrays(np.asarray(spot, dtype=float),
                               np.asarray(variance, dtype=float))
    if np.any(s <= 0) or np.any(v < 0):
        raise ValueError("spot must be positive and variance nonnegative")
    arrays = np.broadcast_arrays(s, v, value, dt, ds, dv, dss, dsv, dvv)
    if any(not np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("inputs must be finite")
    s, v, val, time_d, spot_d, var_d, spot2, cross, var2 = arrays
    return (
        time_d + 0.5 * v * s**2 * spot2 + rho * xi * v * s * cross
        + 0.5 * xi**2 * v * var2 + rate * s * spot_d
        + kappa * (theta - v) * var_d - rate * val
    )
