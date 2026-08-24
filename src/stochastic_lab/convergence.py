"""Convergence diagnostics for stochastic differential equation schemes."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Scheme = Callable[..., tuple[np.ndarray, np.ndarray]]


def gbm_exact_terminal(
    s0: float,
    mu: float,
    sigma: float,
    horizon: float,
    brownian_terminal: np.ndarray,
) -> np.ndarray:
    """Exact terminal value of geometric Brownian motion for supplied W_T."""
    return s0 * np.exp(
        (mu - 0.5 * sigma**2) * horizon + sigma * brownian_terminal
    )


def estimate_loglog_slope(step_sizes: np.ndarray, errors: np.ndarray) -> float:
    """Estimate a convergence order from log(error) against log(step size)."""
    h = np.asarray(step_sizes, dtype=float)
    e = np.asarray(errors, dtype=float)
    if h.ndim != 1 or e.ndim != 1 or h.size != e.size or h.size < 2:
        raise ValueError("step_sizes and errors must be one-dimensional and equal length")
    if np.any(h <= 0) or np.any(e <= 0):
        raise ValueError("step sizes and errors must be strictly positive")
    slope, _ = np.polyfit(np.log(h), np.log(e), 1)
    return float(slope)


def gbm_strong_errors(
    scheme: str,
    step_counts: list[int],
    paths: int = 20_000,
    s0: float = 100.0,
    mu: float = 0.05,
    sigma: float = 0.2,
    horizon: float = 1.0,
    seed: int = 1234,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Estimate terminal strong error for Euler-Maruyama or Milstein on GBM.

    Each resolution is driven by a Brownian path generated from the same RNG seed.
    The exact solution and numerical approximation therefore share the same terminal
    Brownian increment, producing a genuine pathwise strong-error comparison.
    """
    if scheme not in {"euler", "milstein"}:
        raise ValueError("scheme must be 'euler' or 'milstein'")
    if paths <= 0 or not step_counts or any(n <= 0 for n in step_counts):
        raise ValueError("paths and all step counts must be positive")

    rng = np.random.default_rng(seed)
    errors = []
    step_sizes = []
    for steps in step_counts:
        dt = horizon / steps
        dw = np.sqrt(dt) * rng.standard_normal((paths, steps))
        exact = gbm_exact_terminal(s0, mu, sigma, horizon, dw.sum(axis=1))
        numerical = np.full(paths, s0, dtype=float)
        for i in range(steps):
            increment = dw[:, i]
            if scheme == "euler":
                numerical += mu * numerical * dt + sigma * numerical * increment
            else:
                numerical += (
                    mu * numerical * dt
                    + sigma * numerical * increment
                    + 0.5
                    * sigma**2
                    * numerical
                    * (increment**2 - dt)
                )
        errors.append(np.mean(np.abs(exact - numerical)))
        step_sizes.append(dt)

    h = np.asarray(step_sizes)
    err = np.asarray(errors)
    return h, err, estimate_loglog_slope(h, err)


def gbm_weak_errors(
    scheme: str,
    step_counts: list[int],
    paths: int = 100_000,
    s0: float = 100.0,
    mu: float = 0.05,
    sigma: float = 0.2,
    horizon: float = 1.0,
    seed: int = 4321,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Estimate weak error in E[S_T] for Euler-Maruyama or Milstein on GBM."""
    if scheme not in {"euler", "milstein"}:
        raise ValueError("scheme must be 'euler' or 'milstein'")
    target = s0 * np.exp(mu * horizon)
    rng = np.random.default_rng(seed)
    errors = []
    step_sizes = []

    for steps in step_counts:
        dt = horizon / steps
        numerical = np.full(paths, s0, dtype=float)
        for _ in range(steps):
            dw = np.sqrt(dt) * rng.standard_normal(paths)
            if scheme == "euler":
                numerical += mu * numerical * dt + sigma * numerical * dw
            else:
                numerical += (
                    mu * numerical * dt
                    + sigma * numerical * dw
                    + 0.5 * sigma**2 * numerical * (dw**2 - dt)
                )
        errors.append(abs(numerical.mean() - target))
        step_sizes.append(dt)

    h = np.asarray(step_sizes)
    err = np.maximum(np.asarray(errors), np.finfo(float).eps)
    return h, err, estimate_loglog_slope(h, err)
