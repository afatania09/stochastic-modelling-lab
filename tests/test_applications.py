import numpy as np

from stochastic_lab.pricing import (
    black_scholes_call,
    european_call_mc,
    european_call_mc_control_variate,
)
from stochastic_lab.risk import expected_shortfall, maximum_drawdown, value_at_risk


def test_mc_call_agrees_with_black_scholes():
    analytic = black_scholes_call(100.0, 100.0, 0.03, 0.20, 1.0)
    estimate, se = european_call_mc(100.0, 100.0, 0.03, 0.20, 1.0, 150_000, 123)
    assert abs(estimate - analytic) < 4.0 * se


def test_control_variate_reduces_pricing_standard_error():
    _, plain_se = european_call_mc(100.0, 100.0, 0.03, 0.20, 1.0, 80_000, 9)
    _, cv_se = european_call_mc_control_variate(100.0, 100.0, 0.03, 0.20, 1.0, 80_000, 9)
    assert cv_se < plain_se


def test_expected_shortfall_exceeds_var_for_continuous_tail():
    losses = np.random.default_rng(2).normal(size=100_000)
    var = value_at_risk(losses, 0.975)
    es = expected_shortfall(losses, 0.975)
    assert es > var


def test_maximum_drawdown_known_path():
    path = np.array([100.0, 120.0, 90.0, 95.0, 80.0, 110.0])
    assert np.isclose(maximum_drawdown(path), 1.0 - 80.0 / 120.0)
