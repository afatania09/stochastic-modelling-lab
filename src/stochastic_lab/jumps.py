"""Jump-process and jump-diffusion models."""

from __future__ import annotations

import numpy as np


def compound_poisson_process(
    intensity: float,
    jump_mean: float,
    jump_std: float,
    horizon: float,
    steps: int,
    paths: int = 1,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a compound Poisson process with Gaussian jump sizes."""
    if intensity < 0 or jump_std < 0 or horizon <= 0 or steps <= 0 or paths <= 0:
        raise ValueError("invalid process parameters")
    dt = horizon / steps
    rng = np.random.default_rng(seed)
    counts = rng.poisson(intensity * dt, size=(paths, steps))
    jumps = np.zeros((paths, steps), dtype=float)
    for p in range(paths):
        for i in range(steps):
            n = counts[p, i]
            if n:
                jumps[p, i] = rng.normal(jump_mean, jump_std, size=n).sum()
    values = np.column_stack([np.zeros(paths), np.cumsum(jumps, axis=1)])
    time = np.linspace(0.0, horizon, steps + 1)
    return time, values


def merton_jump_diffusion(
    s0: float,
    mu: float,
    sigma: float,
    jump_intensity: float,
    jump_mean: float,
    jump_std: float,
    horizon: float,
    steps: int,
    paths: int = 1,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate Merton's lognormal jump-diffusion model.

    Jump log-sizes are Gaussian. The drift includes the compensator so that the
    expected asset growth remains approximately exp(mu t) under the chosen measure.
    """
    if s0 <= 0 or sigma < 0 or jump_intensity < 0 or jump_std < 0:
        raise ValueError("invalid model parameters")
    if horizon <= 0 or steps <= 0 or paths <= 0:
        raise ValueError("horizon, steps and paths must be positive")

    dt = horizon / steps
    rng = np.random.default_rng(seed)
    kappa = np.exp(jump_mean + 0.5 * jump_std**2) - 1.0
    log_paths = np.full((paths, steps + 1), np.log(s0), dtype=float)

    for i in range(steps):
        z = rng.standard_normal(paths)
        counts = rng.poisson(jump_intensity * dt, size=paths)
        jump_sums = np.zeros(paths)
        active = np.flatnonzero(counts)
        for p in active:
            jump_sums[p] = rng.normal(jump_mean, jump_std, size=counts[p]).sum()
        log_paths[:, i + 1] = (
            log_paths[:, i]
            + (mu - jump_intensity * kappa - 0.5 * sigma**2) * dt
            + sigma * np.sqrt(dt) * z
            + jump_sums
        )

    time = np.linspace(0.0, horizon, steps + 1)
    return time, np.exp(log_paths)
