"""Short-rate term-structure models and zero-coupon bond analytics."""

from __future__ import annotations

import numpy as np


def vasicek_paths(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    horizon: float = 1.0,
    steps: int = 252,
    paths: int = 1,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact-discretisation simulation of the Vasicek short-rate model."""
    if kappa <= 0 or sigma < 0 or horizon <= 0 or steps <= 0 or paths <= 0:
        raise ValueError("kappa, horizon, steps and paths must be positive; sigma non-negative")
    dt = horizon / steps
    phi = np.exp(-kappa * dt)
    variance = sigma**2 * (1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((paths, steps))
    rates = np.empty((paths, steps + 1), dtype=float)
    rates[:, 0] = r0
    for i in range(steps):
        rates[:, i + 1] = theta + (rates[:, i] - theta) * phi + np.sqrt(variance) * z[:, i]
    return np.linspace(0.0, horizon, steps + 1), rates


def vasicek_zero_coupon_bond_price(
    short_rate: float | np.ndarray,
    maturity: float,
    kappa: float,
    theta: float,
    sigma: float,
) -> np.ndarray:
    """Analytical Vasicek zero-coupon price P(0,T)=A(T)exp(-B(T)r0)."""
    if maturity < 0 or kappa <= 0 or sigma < 0:
        raise ValueError("maturity and sigma must be non-negative; kappa positive")
    r = np.asarray(short_rate, dtype=float)
    if maturity == 0:
        return np.ones_like(r)
    b = (1.0 - np.exp(-kappa * maturity)) / kappa
    log_a = (
        (theta - sigma**2 / (2.0 * kappa**2)) * (b - maturity)
        - sigma**2 * b**2 / (4.0 * kappa)
    )
    return np.exp(log_a - b * r)


def zero_coupon_yield(price: float | np.ndarray, maturity: float) -> np.ndarray:
    """Continuously compounded yield implied by a zero-coupon bond price."""
    if maturity <= 0:
        raise ValueError("maturity must be positive")
    p = np.asarray(price, dtype=float)
    if np.any(p <= 0):
        raise ValueError("price must be positive")
    return -np.log(p) / maturity


def vasicek_yield_curve(
    short_rate: float,
    maturities: np.ndarray,
    kappa: float,
    theta: float,
    sigma: float,
) -> np.ndarray:
    """Evaluate the analytical Vasicek zero-coupon yield curve."""
    mats = np.asarray(maturities, dtype=float)
    if mats.ndim != 1 or np.any(mats <= 0):
        raise ValueError("maturities must be a one-dimensional positive array")
    prices = np.array([
        vasicek_zero_coupon_bond_price(short_rate, float(t), kappa, theta, sigma)
        for t in mats
    ])
    return -np.log(prices.astype(float)) / mats
