import numpy as np

from stochastic_lab.factor_models import (
    covariance_explained,
    pca_factor_decomposition,
    reconstruct_from_factors,
)
from stochastic_lab.heston_pricing import heston_calibration_rmse, heston_european_call_mc
from stochastic_lab.multivariate import correlated_brownian_motion, correlated_gbm
from stochastic_lab.pricing import black_scholes_call
from stochastic_lab.term_structure import (
    vasicek_paths,
    vasicek_yield_curve,
    vasicek_zero_coupon_bond_price,
    zero_coupon_yield,
)


def test_correlated_brownian_recovers_target_correlation():
    target = np.array([[1.0, 0.65], [0.65, 1.0]])
    _, paths = correlated_brownian_motion(target, horizon=1.0, steps=1, paths=80_000, seed=7)
    terminal = paths[:, :, -1]
    empirical = np.corrcoef(terminal.T)
    assert abs(empirical[0, 1] - 0.65) < 0.02


def test_correlated_gbm_shapes_and_positivity():
    _, values = correlated_gbm(
        np.array([100.0, 80.0]),
        np.array([0.05, 0.03]),
        np.array([0.2, 0.3]),
        np.array([[1.0, -0.25], [-0.25, 1.0]]),
        steps=12,
        paths=100,
        seed=1,
    )
    assert values.shape == (100, 2, 13)
    assert np.all(values > 0)


def test_pca_reconstructs_full_rank_data():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(500, 4)) @ np.array(
        [[1.0, 0.6, 0.2, 0.0], [0.0, 1.0, 0.4, 0.1], [0.0, 0.0, 0.7, 0.2], [0.0, 0.0, 0.0, 0.4]]
    )
    fit = pca_factor_decomposition(x, n_components=4)
    reconstructed = reconstruct_from_factors(fit["scores"], fit["components"], fit["mean"])
    assert np.allclose(reconstructed, x, atol=1e-10)
    assert np.isclose(covariance_explained(fit["eigenvalues"], 4), 1.0)


def test_vasicek_zero_coupon_boundary_and_yield_curve():
    assert np.isclose(vasicek_zero_coupon_bond_price(0.04, 0.0, 1.2, 0.05, 0.015), 1.0)
    maturities = np.array([0.5, 1.0, 2.0, 5.0])
    yields = vasicek_yield_curve(0.03, maturities, 1.2, 0.05, 0.015)
    assert yields.shape == maturities.shape
    assert np.all(np.isfinite(yields))
    one_year_price = vasicek_zero_coupon_bond_price(0.03, 1.0, 1.2, 0.05, 0.015)
    assert np.isclose(zero_coupon_yield(one_year_price, 1.0), yields[1])


def test_vasicek_simulated_long_run_mean():
    _, paths = vasicek_paths(0.01, 1.5, 0.04, 0.01, horizon=10.0, steps=1000, paths=10_000, seed=12)
    assert abs(paths[:, -1].mean() - 0.04) < 0.003


def test_heston_call_reduces_to_near_black_scholes_for_small_vol_of_vol():
    s0, strike, rate, maturity, variance = 100.0, 100.0, 0.02, 1.0, 0.04
    heston_price, se = heston_european_call_mc(
        s0=s0,
        strike=strike,
        rate=rate,
        maturity=maturity,
        v0=variance,
        kappa=5.0,
        theta=variance,
        xi=0.02,
        rho=0.0,
        steps=252,
        paths=40_000,
        seed=99,
    )
    bs = black_scholes_call(s0, strike, rate, np.sqrt(variance), maturity)
    assert abs(heston_price - bs) < max(0.35, 4.0 * se)


def test_heston_calibration_rmse_zero_on_identical_surface():
    prices = np.array([[10.0, 6.0], [14.0, 9.0]])
    assert heston_calibration_rmse(prices, prices) == 0.0
