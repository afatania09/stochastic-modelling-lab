"""Manufactured Heston PDE operator checks."""
import numpy as np
import pytest

from stochastic_lab.heston_operator import heston_backward_residual

PARAMS = {"rate": 0.03, "kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.6}


def residual(**derivatives):
    return heston_backward_residual(
        spot=np.array([80.0, 100.0, 120.0]),
        variance=np.array([0.0, 0.04, 0.2]),
        **PARAMS,
        **derivatives,
    )


def test_discounted_stock_is_exact_solution():
    # V=S is the zero-dividend perpetual stock-price solution:
    # r*S*V_S-r*V=0, with all other derivatives zero.
    s = np.array([80.0, 100.0, 120.0])
    z = np.zeros(3)
    result = residual(value=s, dt=z, ds=np.ones(3), dv=z,
                      dss=z, dsv=z, dvv=z)
    np.testing.assert_allclose(result, 0, atol=1e-12)


def test_constant_discounted_claim():
    # V(t)=exp(-r*(T-t)) satisfies V_t=r*V.
    val = np.full(3, np.exp(-PARAMS["rate"]))
    z = np.zeros(3)
    result = residual(value=val, dt=PARAMS["rate"] * val,
                      ds=z, dv=z, dss=z, dsv=z, dvv=z)
    np.testing.assert_allclose(result, 0, atol=1e-12)


def test_cross_derivative_coefficient():
    # Manufactured V=S*v: V_S=v, V_v=S, V_Sv=1.
    s = np.array([80.0, 100.0, 120.0])
    v = np.array([0.0, 0.04, 0.2])
    z = np.zeros(3)
    actual = residual(value=s * v, dt=z, ds=v, dv=s,
                      dss=z, dsv=np.ones(3), dvv=z)
    expected = PARAMS["rho"] * PARAMS["xi"] * v * s
    expected += PARAMS["kappa"] * (PARAMS["theta"] - v) * s
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_invalid_variance_rejected():
    with pytest.raises(ValueError):
        heston_backward_residual(
            spot=np.array([100.0]), variance=np.array([-0.1]),
            value=0.0, dt=0.0, ds=0.0, dv=0.0, dss=0.0,
            dsv=0.0, dvv=0.0, **PARAMS
        )
