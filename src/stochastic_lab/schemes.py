"""Generic numerical schemes for scalar stochastic differential equations."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Drift = Callable[[float, float], float]
Diffusion = Callable[[float, float], float]


def euler_maruyama(
    x0: float,
    drift: Drift,
    diffusion: Diffusion,
    horizon: float,
    steps: int,
    paths: int = 1,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Euler-Maruyama approximation for dX=a(t,X)dt+b(t,X)dW."""
    if horizon <= 0 or steps <= 0 or paths <= 0:
        raise ValueError("horizon, steps and paths must be positive")
    dt = horizon / steps
    time = np.linspace(0.0, horizon, steps + 1)
    z = np.random.default_rng(seed).standard_normal((paths, steps))
    x = np.empty((paths, steps + 1), dtype=float)
    x[:, 0] = x0
    for i in range(steps):
        t = time[i]
        for p in range(paths):
            xp = x[p, i]
            x[p, i + 1] = (
                xp
                + drift(t, xp) * dt
                + diffusion(t, xp) * np.sqrt(dt) * z[p, i]
            )
    return time, x


def milstein(
    x0: float,
    drift: Drift,
    diffusion: Diffusion,
    diffusion_dx: Diffusion,
    horizon: float,
    steps: int,
    paths: int = 1,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Milstein approximation for scalar SDEs with differentiable diffusion."""
    if horizon <= 0 or steps <= 0 or paths <= 0:
        raise ValueError("horizon, steps and paths must be positive")
    dt = horizon / steps
    time = np.linspace(0.0, horizon, steps + 1)
    z = np.random.default_rng(seed).standard_normal((paths, steps))
    x = np.empty((paths, steps + 1), dtype=float)
    x[:, 0] = x0
    for i in range(steps):
        t = time[i]
        dw = np.sqrt(dt) * z[:, i]
        for p in range(paths):
            xp = x[p, i]
            b = diffusion(t, xp)
            x[p, i + 1] = (
                xp
                + drift(t, xp) * dt
                + b * dw[p]
                + 0.5 * b * diffusion_dx(t, xp) * (dw[p] ** 2 - dt)
            )
    return time, x
