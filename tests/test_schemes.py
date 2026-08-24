import numpy as np

from stochastic_lab import euler_maruyama, milstein


def test_euler_maruyama_reproducible():
    drift = lambda t, x: 0.1 * x
    diffusion = lambda t, x: 0.2 * x
    _, a = euler_maruyama(100.0, drift, diffusion, 1.0, 100, paths=10, seed=42)
    _, b = euler_maruyama(100.0, drift, diffusion, 1.0, 100, paths=10, seed=42)
    assert np.allclose(a, b)


def test_milstein_positive_for_small_gbm_steps():
    mu, sigma = 0.05, 0.2
    drift = lambda t, x: mu * x
    diffusion = lambda t, x: sigma * x
    diffusion_dx = lambda t, x: sigma
    _, x = milstein(100.0, drift, diffusion, diffusion_dx, 1.0, 252, paths=1000, seed=7)
    assert np.all(x > 0.0)
