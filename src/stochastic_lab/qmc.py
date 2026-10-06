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


def brownian_bridge_from_normals(
    normals: np.ndarray,
    horizon: float = 1.0,
) -> np.ndarray:
    """Construct Brownian paths from independent normals using a bridge ordering.

    The first normal determines the terminal value W(T). Subsequent normals
    recursively fill conditional midpoints. This concentrates the largest
    path variance in the earliest coordinates, which is useful with
    low-discrepancy sequences.
    """
    z = np.asarray(normals, dtype=float)
    if z.ndim != 2 or z.shape[1] < 1:
        raise ValueError("normals must have shape (paths, steps) with steps >= 1")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    paths, steps = z.shape
    time = np.linspace(0.0, horizon, steps + 1)
    values = np.empty((paths, steps + 1), dtype=float)
    values[:, 0] = 0.0
    values[:, steps] = np.sqrt(horizon) * z[:, 0]

    intervals: list[tuple[int, int]] = [(0, steps)]
    normal_index = 1
    while intervals:
        left, right = intervals.pop(0)
        if right - left <= 1:
            continue

        middle = (left + right) // 2
        t_left, t_middle, t_right = time[left], time[middle], time[right]
        span = t_right - t_left
        left_weight = (t_right - t_middle) / span
        right_weight = (t_middle - t_left) / span
        conditional_mean = left_weight * values[:, left] + right_weight * values[:, right]
        conditional_variance = (t_middle - t_left) * (t_right - t_middle) / span
        values[:, middle] = (
            conditional_mean + np.sqrt(conditional_variance) * z[:, normal_index]
        )
        normal_index += 1
        intervals.append((left, middle))
        intervals.append((middle, right))

    if normal_index != steps:
        raise RuntimeError("Brownian bridge did not consume the expected number of normals")
    return values


def sobol_brownian_bridge(
    steps: int,
    power: int,
    horizon: float = 1.0,
    scramble: bool = True,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Brownian paths using Sobol normals and Brownian-bridge ordering."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    normals = sobol_normal(
        dimension=steps,
        power=power,
        scramble=scramble,
        seed=seed,
    )
    paths = brownian_bridge_from_normals(normals, horizon=horizon)
    return np.linspace(0.0, horizon, steps + 1), paths


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
