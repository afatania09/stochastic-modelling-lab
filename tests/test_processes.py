import numpy as np

from stochastic_lab import (
    brownian_motion,
    cir_process,
    geometric_brownian_motion,
    ornstein_uhlenbeck,
    poisson_process,
)


def test_brownian_terminal_moments():
    _, w = brownian_motion(horizon=1.0, steps=252, paths=20000, seed=7)
    terminal = w[:, -1]
    assert abs(terminal.mean()) < 0.03
    assert abs(terminal.var(ddof=1) - 1.0) < 0.04


def test_gbm_terminal_mean_matches_theory():
    s0, mu, t = 100.0, 0.06, 1.0
    _, s = geometric_brownian_motion(s0, mu, 0.2, t, 252, 20000, seed=11)
    expected = s0 * np.exp(mu * t)
    assert abs(s[:, -1].mean() / expected - 1.0) < 0.01
    assert np.all(s > 0)


def test_ou_reverts_toward_long_run_mean():
    _, x = ornstein_uhlenbeck(5.0, theta=1.0, kappa=4.0, sigma=0.2, horizon=2.0, steps=500, paths=5000, seed=3)
    assert abs(x[:, -1].mean() - 1.0) < 0.03


def test_cir_is_nonnegative():
    _, x = cir_process(0.03, 0.04, 2.0, 0.2, horizon=2.0, steps=500, paths=1000, seed=5)
    assert np.all(x >= 0.0)


def test_poisson_terminal_mean_matches_lambda_t():
    lam, t = 3.5, 2.0
    _, n = poisson_process(lam, t, steps=400, paths=20000, seed=19)
    assert abs(n[:, -1].mean() - lam * t) < 0.08
