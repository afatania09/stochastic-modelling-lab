import numpy as np

from stochastic_lab.convergence import (
    estimate_loglog_slope,
    gbm_exact_terminal,
    gbm_strong_errors,
)
from stochastic_lab.jumps import compound_poisson_process, merton_jump_diffusion


def test_exact_gbm_terminal_matches_closed_form():
    wt = np.array([0.0, 1.0, -1.0])
    terminal = gbm_exact_terminal(100.0, 0.05, 0.2, 1.0, wt)
    expected = 100.0 * np.exp((0.05 - 0.5 * 0.2**2) + 0.2 * wt)
    assert np.allclose(terminal, expected)


def test_loglog_slope_recovers_known_order():
    h = np.array([0.5, 0.25, 0.125, 0.0625])
    error = 2.0 * h**1.5
    assert abs(estimate_loglog_slope(h, error) - 1.5) < 1e-12


def test_milstein_strong_error_converges_faster_than_euler_for_gbm():
    step_counts = [8, 16, 32, 64]
    _, _, euler_order = gbm_strong_errors(
        "euler", step_counts, paths=6000, seed=100
    )
    _, _, milstein_order = gbm_strong_errors(
        "milstein", step_counts, paths=6000, seed=100
    )
    assert euler_order > 0.25
    assert milstein_order > 0.65
    assert milstein_order > euler_order


def test_compound_poisson_expected_terminal_mean():
    intensity = 3.0
    jump_mean = 0.4
    horizon = 2.0
    _, paths = compound_poisson_process(
        intensity,
        jump_mean,
        0.3,
        horizon,
        steps=100,
        paths=12000,
        seed=7,
    )
    theoretical = intensity * horizon * jump_mean
    assert abs(paths[:, -1].mean() - theoretical) < 0.08


def test_merton_jump_diffusion_preserves_expected_growth_with_compensator():
    s0 = 100.0
    mu = 0.04
    horizon = 1.0
    _, paths = merton_jump_diffusion(
        s0=s0,
        mu=mu,
        sigma=0.2,
        jump_intensity=0.8,
        jump_mean=-0.08,
        jump_std=0.18,
        horizon=horizon,
        steps=120,
        paths=15000,
        seed=17,
    )
    target = s0 * np.exp(mu * horizon)
    assert abs(paths[:, -1].mean() - target) / target < 0.025
