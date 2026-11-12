"""Independent analytical and structural checks for Heston Fourier pricing."""
import numpy as np
import pytest

from stochastic_lab.heston_fourier import (
    heston_characteristic_function,
    heston_european_call_fourier,
)
from stochastic_lab.pricing import black_scholes_call


PARAMS = dict(s0=100.0, rate=0.02, maturity=1.0, v0=0.04,
              kappa=3.0, theta=0.04, xi=0.2, rho=-0.5)


def test_characteristic_function_at_zero():
    assert np.isclose(heston_characteristic_function(0.0, **PARAMS), 1.0)


def test_characteristic_function_first_moment():
    phi = heston_characteristic_function(-1j, **PARAMS)
    assert np.isclose(phi, PARAMS["s0"] * np.exp(PARAMS["rate"] * PARAMS["maturity"]))


def test_call_bounds_and_strike_monotonicity():
    prices = [
        heston_european_call_fourier(strike=k, **PARAMS)
        for k in (80.0, 100.0, 120.0)
    ]
    assert prices[0] >= prices[1] >= prices[2] >= 0.0
    assert all(p <= PARAMS["s0"] for p in prices)


def test_near_black_scholes_when_variance_is_almost_constant():
    params = dict(PARAMS, kappa=5.0, xi=0.01, rho=0.0)
    actual = heston_european_call_fourier(strike=100.0, **params)
    expected = black_scholes_call(100.0, 100.0, 0.02, 0.2, 1.0)
    assert abs(actual - expected) < 0.10


def test_fourier_truncation_stability():
    p100 = heston_european_call_fourier(strike=100.0, **PARAMS, integration_limit=100)
    p200 = heston_european_call_fourier(strike=100.0, **PARAMS, integration_limit=200)
    assert abs(p100 - p200) < 0.01


def test_zero_maturity_and_invalid_parameters():
    assert heston_european_call_fourier(
        strike=90.0, **dict(PARAMS, maturity=0.0)
    ) == 10.0
    with pytest.raises(ValueError):
        heston_european_call_fourier(strike=-1.0, **PARAMS)
