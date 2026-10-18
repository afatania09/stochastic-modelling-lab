"""Independent strong-vs-weak integral consistency tests.

Compare the weak spot formulation with direct quadrature of the strong
Heston operator on manufactured smooth solutions. This is a mathematical
verification test, not an empirical PINN result.
"""
import numpy as np
import pytest

from stochastic_lab.heston_operator import heston_backward_residual
from stochastic_lab.weak_heston import weak_spot_residual
from stochastic_lab.weak_quadrature import gauss_legendre_integral

PARAMS = {"rate": 0.03, "kappa": 1.7, "theta": 0.05, "xi": 0.4, "rho": -0.5}


def polynomial_fields(s, variance):
    # V(S,v)=S**2*v + S*v**2. Mixed and variance derivatives nonzero.
    z = np.zeros_like(s)
    return {
        "value": s**2 * variance + s * variance**2,
        "dt": z,
        "ds": 2 * s * variance + variance**2,
        "dv": s**2 + 2 * s * variance,
        "dss": np.full_like(s, 2 * variance),
        "dsv": 2 * s + 2 * variance,
        "dvv": 2 * s,
    }


@pytest.mark.parametrize("variance", [0.0, 0.04, 0.2])
@pytest.mark.parametrize("order", [12, 24, 48])
def test_weak_matches_direct_strong_operator(variance, order):
    lower, upper = 80.0, 120.0

    def fields(s):
        return polynomial_fields(s, variance)

    def test(s):
        return 1.0 + (s - 100.0) / 40.0

    def test_ds(s):
        return np.full_like(s, 1.0 / 40.0)

    weak = weak_spot_residual(
        lower=lower, upper=upper, variance=variance, order=order,
        fields=fields, test=test, test_ds=test_ds, **PARAMS
    )

    def direct(s):
        f = fields(s)
        strong = heston_backward_residual(
            spot=s, variance=np.full_like(s, variance),
            **PARAMS, **f
        )
        return test(s) * strong

    strong_integral = gauss_legendre_integral(direct, lower, upper, order)
    assert weak == pytest.approx(strong_integral, rel=1e-12, abs=1e-9)
