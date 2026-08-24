import numpy as np

from stochastic_lab.monte_carlo import (
    antithetic_normal_samples,
    control_variate_estimate,
    plain_monte_carlo,
    running_mean_standard_error,
)


def test_antithetic_samples_pair_exactly_for_even_path_count():
    samples = antithetic_normal_samples(paths=1000, dimension=3, seed=123)
    first = samples[:500]
    second = samples[500:]
    assert np.allclose(first, -second)


def test_plain_monte_carlo_estimates_standard_normal_second_moment():
    rng = np.random.default_rng(42)
    z = rng.standard_normal(50000)
    estimate, standard_error = plain_monte_carlo(z, lambda x: x**2)
    assert abs(estimate - 1.0) < 4 * standard_error
    assert standard_error > 0


def test_control_variate_reduces_standard_error():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(30000)
    noise = 0.3 * rng.standard_normal(30000)
    target = 2.0 + 1.5 * x + noise
    plain_se = target.std(ddof=1) / np.sqrt(target.size)
    estimate, adjusted_se, beta = control_variate_estimate(target, x, 0.0)
    assert abs(estimate - 2.0) < 0.02
    assert adjusted_se < plain_se
    assert 1.3 < beta < 1.7


def test_running_mean_standard_error_shapes_and_terminal_mean():
    x = np.arange(1.0, 101.0)
    means, errors = running_mean_standard_error(x)
    assert means.shape == x.shape
    assert errors.shape == x.shape
    assert np.isnan(errors[0])
    assert np.isclose(means[-1], x.mean())
