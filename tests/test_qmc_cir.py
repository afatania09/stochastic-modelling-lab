import numpy as np

from stochastic_lab.qmc import brownian_bridge_from_normals, sobol_brownian_bridge
from stochastic_lab.term_structure import cir_yield_curve, cir_zero_coupon_bond_price


def test_brownian_bridge_recovers_terminal_normal_mapping():
    rng = np.random.default_rng(123)
    normals = rng.standard_normal((20_000, 8))
    paths = brownian_bridge_from_normals(normals, horizon=2.0)

    assert paths.shape == (20_000, 9)
    assert np.all(paths[:, 0] == 0.0)
    assert np.allclose(paths[:, -1], np.sqrt(2.0) * normals[:, 0])


def test_brownian_bridge_matches_brownian_covariance():
    rng = np.random.default_rng(9)
    normals = rng.standard_normal((60_000, 4))
    paths = brownian_bridge_from_normals(normals, horizon=1.0)

    empirical_cov = np.cov(paths[:, 1:].T)
    times = np.linspace(0.25, 1.0, 4)
    target_cov = np.minimum.outer(times, times)

    assert np.max(np.abs(empirical_cov - target_cov)) < 0.02


def test_sobol_brownian_bridge_shapes_and_terminal_moments():
    time, paths = sobol_brownian_bridge(steps=8, power=12, horizon=1.5, seed=7)

    assert time.shape == (9,)
    assert paths.shape == (4096, 9)
    assert abs(paths[:, -1].mean()) < 0.01
    assert abs(paths[:, -1].var(ddof=1) - 1.5) < 0.02


def test_cir_bond_boundary_and_positive_prices():
    assert np.isclose(cir_zero_coupon_bond_price(0.03, 0.0, 1.2, 0.04, 0.15), 1.0)

    maturities = np.array([0.25, 1.0, 2.0, 5.0, 10.0])
    prices = np.array(
        [cir_zero_coupon_bond_price(0.03, t, 1.2, 0.04, 0.15) for t in maturities]
    )
    assert np.all(np.isfinite(prices))
    assert np.all(prices > 0.0)
    assert np.all(prices <= 1.0)


def test_cir_zero_volatility_matches_deterministic_discounting():
    r0 = 0.025
    kappa = 0.8
    theta = 0.04
    maturity = 3.0

    expected_integral = (
        theta * maturity + (r0 - theta) * (1.0 - np.exp(-kappa * maturity)) / kappa
    )
    expected_price = np.exp(-expected_integral)
    actual_price = cir_zero_coupon_bond_price(r0, maturity, kappa, theta, 0.0)

    assert np.isclose(actual_price, expected_price)


def test_cir_yield_curve_is_finite_and_shape_preserving():
    maturities = np.array([0.5, 1.0, 2.0, 5.0])
    yields = cir_yield_curve(0.03, maturities, 1.5, 0.045, 0.12)

    assert yields.shape == maturities.shape
    assert np.all(np.isfinite(yields))
    assert np.all(yields >= 0.0)
