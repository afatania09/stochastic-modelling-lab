import numpy as np
from scipy.stats import norm

from stochastic_lab.qmc import qmc_integrate, sobol_normal, sobol_uniform
from stochastic_lab.rare_events import (
    normal_tail_probability_importance_sampling,
    normal_tail_probability_mc,
)


def test_importance_sampling_matches_normal_tail_and_reduces_error():
    threshold = 4.0
    truth = norm.sf(threshold)
    plain, plain_se = normal_tail_probability_mc(threshold, simulations=200_000, seed=1)
    tilted, tilted_se = normal_tail_probability_importance_sampling(
        threshold,
        simulations=200_000,
        seed=1,
    )
    assert abs(tilted - truth) < 5 * tilted_se
    assert tilted_se < plain_se
    assert plain >= 0.0


def test_sobol_uniform_shape_and_bounds():
    points = sobol_uniform(dimension=3, power=8, seed=42)
    assert points.shape == (256, 3)
    assert np.all((points >= 0.0) & (points <= 1.0))


def test_sobol_normal_has_reasonable_moments():
    samples = sobol_normal(dimension=2, power=12, seed=42)
    assert abs(float(samples.mean())) < 0.03
    assert abs(float(samples.std()) - 1.0) < 0.04


def test_qmc_integrates_polynomial():
    estimate, standard_error = qmc_integrate(
        lambda x: x[:, 0] ** 2 + x[:, 1] ** 2,
        dimension=2,
        power=10,
        replications=8,
        seed=123,
    )
    assert abs(estimate - 2.0 / 3.0) < 5e-4
    assert standard_error < 5e-4
