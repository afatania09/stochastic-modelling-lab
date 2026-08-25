"""Multivariate Gaussian and correlated diffusion utilities."""

from __future__ import annotations

import numpy as np


def validate_correlation_matrix(correlation: np.ndarray, atol: float = 1e-10) -> np.ndarray:
    """Validate and return a symmetric positive-semidefinite correlation matrix."""
    corr = np.asarray(correlation, dtype=float)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("correlation must be square")
    if not np.allclose(corr, corr.T, atol=atol):
        raise ValueError("correlation must be symmetric")
    if not np.allclose(np.diag(corr), 1.0, atol=atol):
        raise ValueError("correlation diagonal must equal one")
    eigvals = np.linalg.eigvalsh(corr)
    if np.min(eigvals) < -atol:
        raise ValueError("correlation must be positive semidefinite")
    return corr


def correlated_brownian_motion(
    correlation: np.ndarray,
    horizon: float = 1.0,
    steps: int = 252,
    paths: int = 1,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate correlated Brownian motions with shape (paths, factors, steps+1)."""
    corr = validate_correlation_matrix(correlation)
    if horizon <= 0 or steps <= 0 or paths <= 0:
        raise ValueError("horizon, steps and paths must be positive")
    dt = horizon / steps
    vals, vecs = np.linalg.eigh(corr)
    root = vecs @ np.diag(np.sqrt(np.clip(vals, 0.0, None))) @ vecs.T
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((paths, steps, corr.shape[0]))
    increments = np.einsum("pti,ji->ptj", z, root) * np.sqrt(dt)
    values = np.concatenate(
        [np.zeros((paths, 1, corr.shape[0])), np.cumsum(increments, axis=1)], axis=1
    )
    return np.linspace(0.0, horizon, steps + 1), np.transpose(values, (0, 2, 1))


def correlated_gbm(
    initial_values: np.ndarray,
    drifts: np.ndarray,
    volatilities: np.ndarray,
    correlation: np.ndarray,
    horizon: float = 1.0,
    steps: int = 252,
    paths: int = 1,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact simulation of multiple correlated GBMs."""
    s0 = np.asarray(initial_values, dtype=float)
    mu = np.asarray(drifts, dtype=float)
    vol = np.asarray(volatilities, dtype=float)
    corr = validate_correlation_matrix(correlation)
    n = corr.shape[0]
    if s0.shape != (n,) or mu.shape != (n,) or vol.shape != (n,):
        raise ValueError("initial_values, drifts and volatilities must match correlation dimension")
    if np.any(s0 <= 0) or np.any(vol < 0):
        raise ValueError("initial values must be positive and volatilities non-negative")
    time, brownian = correlated_brownian_motion(corr, horizon, steps, paths, seed)
    drift = (mu - 0.5 * vol**2)[None, :, None] * time[None, None, :]
    values = s0[None, :, None] * np.exp(drift + vol[None, :, None] * brownian)
    return time, values
