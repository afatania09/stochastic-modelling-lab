import numpy as np

from stochastic_lab.calibration import parameter_bounds_check, relative_rmse, weighted_rmse


def test_weighted_rmse_zero_for_perfect_fit():
    x = np.array([1.0, 2.0, 3.0])
    assert weighted_rmse(x, x) == 0.0


def test_weighted_rmse_respects_weights():
    observed = np.array([1.0, 1.0])
    modelled = np.array([2.0, 1.0])
    equal = weighted_rmse(observed, modelled)
    downweighted = weighted_rmse(observed, modelled, np.array([0.1, 0.9]))
    assert downweighted < equal


def test_relative_rmse_is_scale_aware():
    observed = np.array([10.0, 20.0])
    modelled = np.array([11.0, 22.0])
    assert np.isclose(relative_rmse(observed, modelled), 0.1)


def test_parameter_bounds_check():
    result = parameter_bounds_check(
        {"rho": -0.7, "kappa": 2.0},
        {"rho": (-1.0, 1.0), "kappa": (0.0, 10.0)},
    )
    assert result == {"rho": True, "kappa": True}
