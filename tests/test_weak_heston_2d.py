"""Independent manufactured checks of full spatial weak Heston PDE."""
import numpy as np
import pytest

from stochastic_lab.heston_operator import heston_backward_residual
from stochastic_lab.weak_heston_2d import weak_heston_2d

PARAMS = {"rate": 0.03, "kappa": 1.7, "theta": 0.05, "xi": 0.4, "rho": -0.5}


def fields(s, v):
    # V=S**2*v + S*v**2; first derivatives only in weak implementation.
    return {"value": s**2 * v + s * v**2,
            "dt": np.zeros_like(s),
            "ds": 2 * s * v + v**2,
            "dv": s**2 + 2 * s * v}


def test_function(s, v):
    return 1 + s / 100 + 2 * v


def test_ds(s, v):
    return np.ones_like(s) / 100


def test_dv(s, v):
    return np.full_like(s, 2.0)


@pytest.mark.parametrize("lower_variance", [0.0, 0.01])
@pytest.mark.parametrize("order", [8, 12, 24])
def test_weak_2d_equals_direct_strong_integral(lower_variance, order):
    weak = weak_heston_2d(
        spot_bounds=(80, 120), variance_bounds=(lower_variance, 0.2),
        order=order, fields=fields, test=test_function,
        test_ds=test_ds, test_dv=test_dv, **PARAMS
    )
    # Independent direct strong-form integral, using polynomial derivatives.
    from numpy.polynomial.legendre import leggauss

    nodes, weights = leggauss(order)
    s = 100 + 20 * nodes
    v = (0.2 + lower_variance) / 2 + (0.2 - lower_variance) / 2 * nodes
    ss, vv = np.meshgrid(s, v, indexing="ij")
    strong = heston_backward_residual(
        spot=ss, variance=vv, value=ss**2 * vv + ss * vv**2,
        dt=np.zeros_like(ss), ds=2 * ss * vv + vv**2,
        dv=ss**2 + 2 * ss * vv, dss=2 * vv,
        dsv=2 * ss + 2 * vv, dvv=2 * ss, **PARAMS
    )
    direct = np.sum(weights[:, None] * weights[None, :]
                    * strong * test_function(ss, vv))
    direct *= 20 * (0.2 - lower_variance) / 2
    assert weak == pytest.approx(direct, rel=1e-11, abs=1e-8)


def test_invalid_domain():
    with pytest.raises(ValueError):
        weak_heston_2d(
            spot_bounds=(0, 120), variance_bounds=(0, 0.2),
            order=12, fields=fields, test=test_function,
            test_ds=test_ds, test_dv=test_dv, **PARAMS
        )
