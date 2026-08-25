"""Parameter-estimation utilities for stochastic models."""

from __future__ import annotations

import numpy as np


def estimate_gbm_mle(prices: np.ndarray, dt: float) -> dict[str, float]:
    """Estimate GBM drift and volatility from equally spaced positive prices."""
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1 or prices.size < 3 or np.any(prices <= 0):
        raise ValueError("prices must be a one-dimensional positive array")
    if dt <= 0:
        raise ValueError("dt must be positive")

    log_returns = np.diff(np.log(prices))
    mean_r = float(log_returns.mean())
    var_r = float(log_returns.var(ddof=0))
    sigma = float(np.sqrt(max(var_r / dt, 0.0)))
    mu = float(mean_r / dt + 0.5 * sigma * sigma)
    return {"mu": mu, "sigma": sigma}


def estimate_ou_ols(values: np.ndarray, dt: float) -> dict[str, float]:
    """Estimate OU parameters using the exact AR(1) transition representation."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size < 4:
        raise ValueError("values must be a one-dimensional array with at least four observations")
    if dt <= 0:
        raise ValueError("dt must be positive")

    y = x[1:]
    lag = x[:-1]
    design = np.column_stack([np.ones_like(lag), lag])
    intercept, phi = np.linalg.lstsq(design, y, rcond=None)[0]
    phi = float(np.clip(phi, 1e-12, 1.0 - 1e-12))
    kappa = float(-np.log(phi) / dt)
    theta = float(intercept / (1.0 - phi))

    residuals = y - (intercept + phi * lag)
    residual_var = float(np.mean(residuals**2))
    sigma2 = residual_var * 2.0 * kappa / max(1.0 - phi**2, 1e-15)
    sigma = float(np.sqrt(max(sigma2, 0.0)))
    return {"kappa": kappa, "theta": theta, "sigma": sigma, "phi": phi}


def estimation_error(estimate: dict[str, float], truth: dict[str, float]) -> dict[str, float]:
    """Return absolute parameter errors for common keys."""
    keys = estimate.keys() & truth.keys()
    return {key: abs(float(estimate[key]) - float(truth[key])) for key in sorted(keys)}
