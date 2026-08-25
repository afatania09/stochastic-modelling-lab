"""Simple derivative-pricing applications built on stochastic simulation."""

from __future__ import annotations

from math import exp, log, sqrt

import numpy as np
from scipy.stats import norm

from .processes import geometric_brownian_motion


def black_scholes_call(s0: float, strike: float, rate: float, sigma: float, maturity: float) -> float:
    """Analytical Black-Scholes European call price without dividends."""
    if min(s0, strike, sigma, maturity) <= 0:
        raise ValueError("s0, strike, sigma and maturity must be positive")
    d1 = (log(s0 / strike) + (rate + 0.5 * sigma**2) * maturity) / (sigma * sqrt(maturity))
    d2 = d1 - sigma * sqrt(maturity)
    return float(s0 * norm.cdf(d1) - strike * exp(-rate * maturity) * norm.cdf(d2))


def european_call_mc(
    s0: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    simulations: int = 100_000,
    seed: int | None = None,
) -> tuple[float, float]:
    """Price a European call under risk-neutral GBM and return price and standard error."""
    if simulations <= 1:
        raise ValueError("simulations must exceed one")
    _, paths = geometric_brownian_motion(
        s0=s0,
        mu=rate,
        sigma=sigma,
        horizon=maturity,
        steps=1,
        paths=simulations,
        seed=seed,
    )
    payoffs = np.maximum(paths[:, -1] - strike, 0.0) * exp(-rate * maturity)
    return float(payoffs.mean()), float(payoffs.std(ddof=1) / sqrt(simulations))


def european_call_mc_control_variate(
    s0: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    simulations: int = 100_000,
    seed: int | None = None,
) -> tuple[float, float]:
    """European call Monte Carlo with discounted stock as a control variate."""
    if simulations <= 1:
        raise ValueError("simulations must exceed one")
    _, paths = geometric_brownian_motion(
        s0=s0,
        mu=rate,
        sigma=sigma,
        horizon=maturity,
        steps=1,
        paths=simulations,
        seed=seed,
    )
    terminal = paths[:, -1]
    discount = exp(-rate * maturity)
    payoff = discount * np.maximum(terminal - strike, 0.0)
    control = discount * terminal
    covariance = np.cov(payoff, control, ddof=1)[0, 1]
    beta = covariance / np.var(control, ddof=1)
    adjusted = payoff - beta * (control - s0)
    return float(adjusted.mean()), float(adjusted.std(ddof=1) / sqrt(simulations))
