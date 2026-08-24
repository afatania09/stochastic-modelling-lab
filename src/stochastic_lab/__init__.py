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
from .qmc import qmc_integrate, sobol_normal, sobol_uniform
from .rare_events import (
    lognormal_loss_exceedance_importance_sampling,
    normal_tail_probability_importance_sampling,
    normal_tail_probability_mc,
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
    "lognormal_loss_exceedance_importance_sampling",
    "merton_jump_diffusion",
    "milstein",
    "normal_tail_probability_importance_sampling",
    "normal_tail_probability_mc",
    "ornstein_uhlenbeck",
    "plain_monte_carlo",
    "poisson_process",
    "qmc_integrate",
    "running_mean_standard_error",
    "sobol_normal",
    "sobol_uniform",
]
