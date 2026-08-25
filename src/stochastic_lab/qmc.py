"""Quasi-Monte Carlo utilities based on Sobol low-discrepancy sequences."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, qmc


def sobol_uniform(
    dimension: int,
    power: int,
    scramble: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """Generate 2**power Sobol points on the unit hypercube."""
    if dimension <= 0 or power < 0:
        raise ValueError("dimension must be positive and power non-negative")
    engine = qmc.Sobol(d=dimension, scramble=scramble, seed=seed)
    return engine.random_base2(m=power)


def sobol_normal(
    dimension: int,
    power: int,
    scramble: bool = True,
    seed: int | None = None,
    clip: float = 1e-12,
) -> np.ndarray:
    """Transform Sobol points to independent standard-normal variates."""
    if not 0 < clip < 0.5:
        raise ValueError("clip must lie in (0, 0.5)")
    uniforms = sobol_uniform(dimension, power, scramble=scramble, seed=seed)
    return norm.ppf(np.clip(uniforms, clip, 1.0 - clip))


def qmc_integrate(
    integrand,
    dimension: int,
    power: int,
    replications: int = 8,
    seed: int | None = None,
) -> tuple[float, float]:
    """Randomised-QMC estimate and replication-based standard error.

    The integrand receives an array with shape (n_points, dimension) and must
    return one value per point. Independent Owen-style scrambles provide a
    practical error estimate across replications.
    """
    if replications <= 1:
        raise ValueError("replications must exceed one")

    seed_sequence = np.random.SeedSequence(seed)
    child_seeds = seed_sequence.spawn(replications)
    estimates = np.empty(replications, dtype=float)

    for i, child in enumerate(child_seeds):
        points = sobol_uniform(
            dimension=dimension,
            power=power,
            scramble=True,
            seed=int(child.generate_state(1)[0]),
        )
        values = np.asarray(integrand(points), dtype=float)
        if values.shape != (points.shape[0],):
            raise ValueError("integrand must return one value per Sobol point")
        estimates[i] = values.mean()

    return float(estimates.mean()), float(estimates.std(ddof=1) / np.sqrt(replications))
