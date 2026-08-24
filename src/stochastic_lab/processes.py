"""Core stochastic processes with reproducible vectorised simulation."""

from __future__ import annotations

import numpy as np


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def brownian_motion(
    horizon: float = 1.0,
    steps: int = 252,
    paths: int = 1,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate standard Brownian motion W_t on an equally spaced grid."""
    if horizon <= 0 or steps <= 0 or paths <= 0:
        raise ValueError("horizon, steps and paths must be positive")
    dt = horizon / steps
    increments = _rng(seed).normal(0.0, np.sqrt(dt), size=(paths, steps))
    values = np.concatenate([np.zeros((paths, 1)), np.cumsum(increments, axis=1)], axis=1)
    time = np.linspace(0.0, horizon, steps + 1)
    return time, values


def geometric_brownian_motion(
    s0: float,
    mu: float,
    sigma: float,
    horizon: float = 1.0,
    steps: int = 252,
    paths: int = 1,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact simulation of dS = mu*S dt + sigma*S dW."""
    if s0 <= 0 or sigma < 0:
        raise ValueError("s0 must be positive and sigma non-negative")
    time, w = brownian_motion(horizon, steps, paths, seed)
    drift = (mu - 0.5 * sigma**2) * time
    values = s0 * np.exp(drift[None, :] + sigma * w)
    return time, values


def ornstein_uhlenbeck(
    x0: float,
    theta: float,
    kappa: float,
    sigma: float,
    horizon: float = 1.0,
    steps: int = 252,
    paths: int = 1,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact discretisation of the Ornstein-Uhlenbeck mean-reverting process."""
    if kappa <= 0 or sigma < 0:
        raise ValueError("kappa must be positive and sigma non-negative")
    if horizon <= 0 or steps <= 0 or paths <= 0:
        raise ValueError("horizon, steps and paths must be positive")
    dt = horizon / steps
    phi = np.exp(-kappa * dt)
    innovation_sd = sigma * np.sqrt((1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa))
    z = _rng(seed).standard_normal((paths, steps))
    x = np.empty((paths, steps + 1), dtype=float)
    x[:, 0] = x0
    for i in range(steps):
        x[:, i + 1] = theta + (x[:, i] - theta) * phi + innovation_sd * z[:, i]
    return np.linspace(0.0, horizon, steps + 1), x


def cir_process(
    x0: float,
    theta: float,
    kappa: float,
    sigma: float,
    horizon: float = 1.0,
    steps: int = 252,
    paths: int = 1,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Full-truncation Euler simulation of the CIR square-root process."""
    if min(x0, theta, kappa, sigma) < 0 or kappa == 0:
        raise ValueError("x0, theta, sigma must be non-negative and kappa positive")
    if horizon <= 0 or steps <= 0 or paths <= 0:
        raise ValueError("horizon, steps and paths must be positive")
    dt = horizon / steps
    z = _rng(seed).standard_normal((paths, steps))
    x = np.empty((paths, steps + 1), dtype=float)
    x[:, 0] = x0
    for i in range(steps):
        xp = np.maximum(x[:, i], 0.0)
        x[:, i + 1] = x[:, i] + kappa * (theta - xp) * dt + sigma * np.sqrt(xp * dt) * z[:, i]
        x[:, i + 1] = np.maximum(x[:, i + 1], 0.0)
    return np.linspace(0.0, horizon, steps + 1), x


def poisson_process(
    intensity: float,
    horizon: float = 1.0,
    steps: int = 252,
    paths: int = 1,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a homogeneous Poisson counting process."""
    if intensity < 0 or horizon <= 0 or steps <= 0 or paths <= 0:
        raise ValueError("intensity must be non-negative; horizon, steps and paths positive")
    dt = horizon / steps
    jumps = _rng(seed).poisson(intensity * dt, size=(paths, steps))
    counts = np.concatenate([np.zeros((paths, 1), dtype=int), np.cumsum(jumps, axis=1)], axis=1)
    return np.linspace(0.0, horizon, steps + 1), counts
