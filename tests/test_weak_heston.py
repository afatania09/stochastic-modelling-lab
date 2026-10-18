"""Manufactured verification of the spot-integrated weak Heston residual."""
import numpy as np
import pytest

from stochastic_lab.weak_heston import weak_spot_residual

PARAMS = {"rate": 0.03, "kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.6}


def call(fields, test, test_ds, variance=0.04, order=12):
    return weak_spot_residual(
        lower=80.0, upper=120.0, variance=variance, order=order,
        fields=fields, test=test, test_ds=test_ds, **PARAMS
    )


def stock_fields(s):
    z = np.zeros_like(s)
    return {"value": s, "dt": z, "ds": np.ones_like(s),
            "dv": z, "dsv": z, "dvv": z}


def quadratic_fields(s):
    # V=S**2. Its strong PDE residual is v*S**2+r*S**2.
    z = np.zeros_like(s)
    return {"value": s**2, "dt": z, "ds": 2 * s,
            "dv": z, "dsv": z, "dvv": z}


def test_exact_stock_solution_nonzero_boundary_test():
    assert call(stock_fields, lambda s: np.ones_like(s),
                lambda s: np.zeros_like(s)) == pytest.approx(0, abs=1e-10)


def test_quadratic_matches_analytic_strong_residual():
    variance = 0.04
    actual = call(quadratic_fields, lambda s: np.ones_like(s),
                  lambda s: np.zeros_like(s), variance=variance)
    expected = (variance + PARAMS["rate"]) * (120**3 - 80**3) / 3
    assert actual == pytest.approx(expected, rel=1e-12)


def test_quadratic_with_nonconstant_test_function():
    # phi=S; integral of (v+r)*S**3.
    variance = 0.08
    actual = call(quadratic_fields, lambda s: s, lambda s: np.ones_like(s),
                  variance=variance)
    expected = (variance + PARAMS["rate"]) * (120**4 - 80**4) / 4
    assert actual == pytest.approx(expected, rel=1e-12)


def test_invalid_bounds_rejected():
    with pytest.raises(ValueError):
        weak_spot_residual(
            lower=0, upper=120, variance=0.04, order=12,
            fields=stock_fields, test=np.ones_like,
            test_ds=np.zeros_like, **PARAMS
        )
