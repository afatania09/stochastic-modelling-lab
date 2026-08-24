"""Monte Carlo estimators and variance-reduction utilities."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Payoff = Callable[[np.ndarray], np.ndarray]


def plain_monte_carlo(
    samples: np.ndarray,
    payoff: Payoff,
) -> tuple[float, float]:
    """Return estimate and standard error from Monte Carlo samples."""
    values = np.asarray(payoff(np.asarray(samples)), dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("payoff must produce at least two scalar observations")
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(values.size))


def antithetic_normal_samples(
    paths: int,
    dimension: int = 1,
    seed: int | None = None,
) -> np.ndarray:
    """Generate standard-normal samples paired with their antithetic negatives."""
    if paths < 2 or dimension <= 0:
        raise ValueError("paths must be at least two and dimension positive")
    half = (paths + 1) // 2
    z = np.random.default_rng(seed).standard_normal((half, dimension))
    paired = np.concatenate([z, -z], axis=0)
    return paired[:paths]


def control_variate_estimate(
    target: np.ndarray,
    control: np.ndarray,
    control_mean: float,
) -> tuple[float, float, float]:
    """Optimal linear control-variate estimate.

    Returns estimate, standard error and fitted control coefficient beta.
    """
    y = np.asarray(target, dtype=float)
    x = np.asarray(control, dtype=float)
    if y.ndim != 1 or x.ndim != 1 or y.size != x.size or y.size < 2:
        raise ValueError("target and control must be equal-length one-dimensional arrays")
    variance = np.var(x, ddof=1)
    if variance <= 0:
        raise ValueError("control must have positive variance")
    beta = np.cov(y, x, ddof=1)[0, 1] / variance
    adjusted = y - beta * (x - control_mean)
    estimate = adjusted.mean()
    standard_error = adjusted.std(ddof=1) / np.sqrt(adjusted.size)
    return float(estimate), float(standard_error), float(beta)


def running_mean_standard_error(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Running Monte Carlo mean and standard error for convergence diagnostics."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("values must contain at least two observations")
    n = np.arange(1, x.size + 1)
    means = np.cumsum(x) / n
    cumulative_sq = np.cumsum(x**2)
    variance = np.zeros_like(x, dtype=float)
    variance[1:] = (cumulative_sq[1:] - n[1:] * means[1:] ** 2) / (n[1:] - 1)
    standard_errors = np.zeros_like(x, dtype=float)
    standard_errors[1:] = np.sqrt(np.maximum(variance[1:], 0.0) / n[1:])
    standard_errors[0] = np.nan
    return means, standard_errors
