"""Reusable stochastic-process simulation tools."""

from .convergence import (
    estimate_loglog_slope,
    gbm_exact_terminal,
    gbm_strong_errors,
    gbm_weak_errors,
)
from .jumps import compound_poisson_process, merton_jump_diffusion
from .monte_carlo import (
    antithetic_normal_samples,
    control_variate_estimate,
    plain_monte_carlo,
    running_mean_standard_error,
)
from .processes import (
    brownian_motion,
    cir_process,
    geometric_brownian_motion,
    ornstein_uhlenbeck,
    poisson_process,
)
from .schemes import euler_maruyama, milstein

__all__ = [
    "antithetic_normal_samples",
    "brownian_motion",
    "cir_process",
    "compound_poisson_process",
    "control_variate_estimate",
    "estimate_loglog_slope",
    "euler_maruyama",
    "gbm_exact_terminal",
    "gbm_strong_errors",
    "gbm_weak_errors",
    "geometric_brownian_motion",
    "merton_jump_diffusion",
    "milstein",
    "ornstein_uhlenbeck",
    "plain_monte_carlo",
    "poisson_process",
    "running_mean_standard_error",
]
