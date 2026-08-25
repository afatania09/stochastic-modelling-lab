import numpy as np

from stochastic_lab.estimation import estimate_gbm_mle, estimate_ou_ols
from stochastic_lab.heston import heston_feller_margin, heston_paths
from stochastic_lab.processes import geometric_brownian_motion, ornstein_uhlenbeck
from stochastic_lab.regime_switching import (
    markov_switching_returns,
    simulate_markov_chain,
    stationary_distribution,
)


def test_gbm_mle_recovers_parameters_reasonably():
    mu, sigma = 0.07, 0.22
    _, paths = geometric_brownian_motion(100.0, mu, sigma, 8.0, 8 * 252, 1, 123)
    estimate = estimate_gbm_mle(paths[0], 1.0 / 252.0)
    assert abs(estimate["sigma"] - sigma) < 0.03
    assert abs(estimate["mu"] - mu) < 0.08


def test_ou_estimator_recovers_mean_reversion():
    kappa, theta, sigma = 1.4, 0.03, 0.18
    _, paths = ornstein_uhlenbeck(0.4, kappa, theta, sigma, 20.0, 10_000, 1, 77)
    estimate = estimate_ou_ols(paths[0], 20.0 / 10_000.0)
    assert abs(estimate["theta"] - theta) < 0.08
    assert estimate["kappa"] > 0
    assert estimate["sigma"] > 0


def test_heston_paths_are_positive_and_variance_nonnegative():
    _, prices, variances = heston_paths(
        100.0, 0.04, 0.05, 2.0, 0.04, 0.35, -0.7, 1.0, 252, 500, 42
    )
    assert np.all(prices > 0)
    assert np.all(variances >= 0)
    assert prices.shape == variances.shape == (500, 253)


def test_heston_feller_margin():
    assert heston_feller_margin(2.0, 0.04, 0.3) > 0
    assert heston_feller_margin(1.0, 0.02, 0.5) < 0


def test_stationary_distribution_is_fixed_point():
    transition = np.array([[0.95, 0.05], [0.15, 0.85]])
    pi = stationary_distribution(transition)
    assert np.allclose(pi @ transition, pi, atol=1e-10)
    assert np.isclose(pi.sum(), 1.0)


def test_markov_chain_long_run_frequency_near_stationary():
    transition = np.array([[0.97, 0.03], [0.10, 0.90]])
    states = simulate_markov_chain(transition, 20_000, 1, 0, 15)[0]
    empirical = np.bincount(states[2_000:], minlength=2) / len(states[2_000:])
    target = stationary_distribution(transition)
    assert np.max(np.abs(empirical - target)) < 0.04


def test_switching_returns_have_regime_dependent_volatility():
    transition = np.array([[0.96, 0.04], [0.08, 0.92]])
    states, returns = markov_switching_returns(
        transition,
        means=np.array([0.0003, -0.0005]),
        volatilities=np.array([0.008, 0.03]),
        steps=20_000,
        seed=55,
    )
    labels = states[0, 1:]
    r = returns[0]
    assert r[labels == 1].std() > 2.0 * r[labels == 0].std()
