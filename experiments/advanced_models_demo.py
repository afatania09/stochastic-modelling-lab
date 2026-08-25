"""Reproducible advanced stochastic-modelling demonstration.

Run with:
    python experiments/advanced_models_demo.py
"""

from __future__ import annotations

import numpy as np

from stochastic_lab import (
    estimate_gbm_mle,
    geometric_brownian_motion,
    heston_feller_margin,
    heston_paths,
    markov_switching_returns,
    stationary_distribution,
)


def main() -> None:
    _, gbm = geometric_brownian_motion(
        s0=100.0,
        mu=0.06,
        sigma=0.20,
        horizon=5.0,
        steps=5 * 252,
        paths=1,
        seed=7,
    )
    estimate = estimate_gbm_mle(gbm[0], 1.0 / 252.0)
    print("GBM MLE:", estimate)

    _, prices, variances = heston_paths(
        s0=100.0,
        v0=0.04,
        mu=0.05,
        kappa=2.0,
        theta=0.04,
        xi=0.30,
        rho=-0.7,
        horizon=1.0,
        steps=252,
        paths=20_000,
        seed=11,
    )
    print("Heston mean terminal price:", float(prices[:, -1].mean()))
    print("Heston mean terminal variance:", float(variances[:, -1].mean()))
    print("Feller margin:", heston_feller_margin(2.0, 0.04, 0.30))

    transition = np.array([[0.97, 0.03], [0.10, 0.90]])
    states, returns = markov_switching_returns(
        transition,
        means=np.array([0.0004, -0.0008]),
        volatilities=np.array([0.008, 0.03]),
        steps=10_000,
        paths=1,
        seed=21,
    )
    print("Stationary regime probabilities:", stationary_distribution(transition))
    print("Simulated return volatility:", float(returns.std()))
    print("Fraction in stress regime:", float(np.mean(states[:, 1:] == 1)))


if __name__ == "__main__":
    main()
