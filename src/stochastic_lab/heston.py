"""Heston stochastic-volatility simulation and diagnostics."""

from __future__ import annotations

import numpy as np


def heston_paths(
    s0: float,
    v0: float,
    mu: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    horizon: float,
    steps: int,
    paths: int = 1,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate Heston price and variance paths using full-truncation Euler."""
    if min(s0, v0, kappa, theta, xi, horizon) <= 0:
        raise ValueError("s0, v0, kappa, theta, xi and horizon must be positive")
    if steps <= 0 or paths <= 0:
        raise ValueError("steps and paths must be positive")
    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [-1, 1]")

    rng = np.random.default_rng(seed)
    dt = horizon / steps
    sqrt_dt = np.sqrt(dt)
    time = np.linspace(0.0, horizon, steps + 1)
    prices = np.empty((paths, steps + 1), dtype=float)
    variances = np.empty_like(prices)
    prices[:, 0] = s0
    variances[:, 0] = v0

    z1 = rng.standard_normal((paths, steps))
    z2_ind = rng.standard_normal((paths, steps))
    z2 = rho * z1 + np.sqrt(max(1.0 - rho * rho, 0.0)) * z2_ind

    for i in range(steps):
        v_pos = np.maximum(variances[:, i], 0.0)
        variances[:, i + 1] = (
            variances[:, i]
            + kappa * (theta - v_pos) * dt
            + xi * np.sqrt(v_pos) * sqrt_dt * z2[:, i]
        )
        variances[:, i + 1] = np.maximum(variances[:, i + 1], 0.0)
        prices[:, i + 1] = prices[:, i] * np.exp(
            (mu - 0.5 * v_pos) * dt + np.sqrt(v_pos) * sqrt_dt * z1[:, i]
        )
    return time, prices, variances


def heston_feller_margin(kappa: float, theta: float, xi: float) -> float:
    """Return 2*kappa*theta - xi^2; positive values satisfy the Feller condition."""
    return float(2.0 * kappa * theta - xi * xi)


def realised_variance(log_prices: np.ndarray) -> float:
    """Return sum of squared log returns."""
    x = np.asarray(log_prices, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("log_prices must be one-dimensional with at least two observations")
    returns = np.diff(x)
    return float(np.sum(returns**2))
