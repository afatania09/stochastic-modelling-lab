"""Monte Carlo pricing and calibration diagnostics for the Heston model."""

from __future__ import annotations

import numpy as np

from .heston import heston_paths


def heston_european_call_mc(
    s0: float,
    strike: float,
    rate: float,
    maturity: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    steps: int = 252,
    paths: int = 50_000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Price a European call under risk-neutral Heston dynamics by Monte Carlo."""
    if strike <= 0 or maturity <= 0:
        raise ValueError("strike and maturity must be positive")
    _, prices, _ = heston_paths(
        s0=s0,
        v0=v0,
        mu=rate,
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=rho,
        horizon=maturity,
        steps=steps,
        paths=paths,
        seed=seed,
    )
    discounted = np.exp(-rate * maturity) * np.maximum(prices[:, -1] - strike, 0.0)
    return float(discounted.mean()), float(discounted.std(ddof=1) / np.sqrt(paths))


def heston_price_surface_mc(
    s0: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    rate: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    steps_per_year: int = 252,
    paths: int = 20_000,
    seed: int = 123,
) -> np.ndarray:
    """Generate a strike-by-maturity Heston call-price surface using common seeds."""
    k = np.asarray(strikes, dtype=float)
    t = np.asarray(maturities, dtype=float)
    if k.ndim != 1 or t.ndim != 1 or np.any(k <= 0) or np.any(t <= 0):
        raise ValueError("strikes and maturities must be positive one-dimensional arrays")
    surface = np.empty((len(t), len(k)), dtype=float)
    for i, maturity in enumerate(t):
        steps = max(1, int(round(steps_per_year * float(maturity))))
        _, simulated, _ = heston_paths(
            s0, v0, rate, kappa, theta, xi, rho, float(maturity), steps, paths, seed + i
        )
        terminal = simulated[:, -1]
        disc = np.exp(-rate * float(maturity))
        surface[i] = [disc * np.maximum(terminal - strike, 0.0).mean() for strike in k]
    return surface


def heston_calibration_rmse(
    model_prices: np.ndarray,
    market_prices: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Weighted RMSE between Heston model and market price surfaces."""
    model = np.asarray(model_prices, dtype=float)
    market = np.asarray(market_prices, dtype=float)
    if model.shape != market.shape or model.size == 0:
        raise ValueError("model_prices and market_prices must share a non-empty shape")
    residual = model - market
    if weights is None:
        return float(np.sqrt(np.mean(residual**2)))
    w = np.asarray(weights, dtype=float)
    if w.shape != model.shape or np.any(w < 0) or np.sum(w) <= 0:
        raise ValueError("weights must match prices, be non-negative and have positive sum")
    return float(np.sqrt(np.sum(w * residual**2) / np.sum(w)))
