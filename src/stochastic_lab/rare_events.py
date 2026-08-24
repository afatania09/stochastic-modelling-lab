"""Rare-event probability estimation with importance sampling."""

from __future__ import annotations

from math import exp, pi, sqrt

import numpy as np


def _normal_pdf(x: np.ndarray, mean: float = 0.0) -> np.ndarray:
    """Standard-deviation-one normal density evaluated at x."""
    return np.exp(-0.5 * (x - mean) ** 2) / sqrt(2.0 * pi)


def normal_tail_probability_mc(
    threshold: float,
    simulations: int = 100_000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Estimate P[Z > threshold] with plain Monte Carlo for Z~N(0,1)."""
    if simulations <= 1:
        raise ValueError("simulations must exceed one")
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal(simulations)
    indicators = (samples > threshold).astype(float)
    estimate = float(indicators.mean())
    standard_error = float(indicators.std(ddof=1) / sqrt(simulations))
    return estimate, standard_error


def normal_tail_probability_importance_sampling(
    threshold: float,
    shift: float | None = None,
    simulations: int = 100_000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Estimate a Gaussian upper-tail probability by exponential tilting.

    Samples are drawn from N(shift, 1) and reweighted by the likelihood ratio
    phi(x) / phi(x-shift). A shift near the threshold is effective for rare
    upper-tail events.
    """
    if simulations <= 1:
        raise ValueError("simulations must exceed one")
    if shift is None:
        shift = max(float(threshold), 0.0)

    rng = np.random.default_rng(seed)
    samples = rng.normal(loc=shift, scale=1.0, size=simulations)
    weights = np.exp(-shift * samples + 0.5 * shift * shift)
    weighted_indicators = (samples > threshold) * weights
    estimate = float(weighted_indicators.mean())
    standard_error = float(weighted_indicators.std(ddof=1) / sqrt(simulations))
    return estimate, standard_error


def lognormal_loss_exceedance_importance_sampling(
    threshold: float,
    mu: float,
    sigma: float,
    shift: float,
    simulations: int = 100_000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Estimate P[exp(mu + sigma Z) > threshold] using tilted normal sampling."""
    if threshold <= 0 or sigma <= 0:
        raise ValueError("threshold and sigma must be positive")
    if simulations <= 1:
        raise ValueError("simulations must exceed one")

    rng = np.random.default_rng(seed)
    z = rng.normal(loc=shift, scale=1.0, size=simulations)
    losses = np.exp(mu + sigma * z)
    likelihood_ratio = np.exp(-shift * z + 0.5 * shift * shift)
    weighted = (losses > threshold) * likelihood_ratio
    return float(weighted.mean()), float(weighted.std(ddof=1) / sqrt(simulations))
