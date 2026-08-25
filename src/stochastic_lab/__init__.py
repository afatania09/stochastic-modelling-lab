"""Reusable stochastic-process simulation tools."""

from .calibration import parameter_bounds_check, relative_rmse, weighted_rmse
from .convergence import (
    estimate_loglog_slope,
    gbm_exact_terminal,
    gbm_strong_errors,
    gbm_weak_errors,
)
from .estimation import estimate_gbm_mle, estimate_ou_ols, estimation_error
from .heston import heston_feller_margin, heston_paths, realised_variance
from .jumps import compound_poisson_process, merton_jump_diffusion
from .monte_carlo import (
    antithetic_normal_samples,
    control_variate_estimate,
    plain_monte_carlo,
    running_mean_standard_error,
)
from .pricing import black_scholes_call, european_call_mc, european_call_mc_control_variate
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
from .regime_switching import (
    markov_switching_returns,
    simulate_markov_chain,
    stationary_distribution,
    validate_transition_matrix,
)
from .risk import drawdown, expected_shortfall, maximum_drawdown, value_at_risk
from .schemes import euler_maruyama, milstein

__all__ = [
    "antithetic_normal_samples",
    "black_scholes_call",
    "brownian_motion",
    "cir_process",
    "compound_poisson_process",
    "control_variate_estimate",
    "drawdown",
    "estimate_gbm_mle",
    "estimate_loglog_slope",
    "estimate_ou_ols",
    "estimation_error",
    "euler_maruyama",
    "european_call_mc",
    "european_call_mc_control_variate",
    "expected_shortfall",
    "gbm_exact_terminal",
    "gbm_strong_errors",
    "gbm_weak_errors",
    "geometric_brownian_motion",
    "heston_feller_margin",
    "heston_paths",
    "lognormal_loss_exceedance_importance_sampling",
    "markov_switching_returns",
    "maximum_drawdown",
    "merton_jump_diffusion",
    "milstein",
    "normal_tail_probability_importance_sampling",
    "normal_tail_probability_mc",
    "ornstein_uhlenbeck",
    "parameter_bounds_check",
    "plain_monte_carlo",
    "poisson_process",
    "qmc_integrate",
    "realised_variance",
    "relative_rmse",
    "running_mean_standard_error",
    "simulate_markov_chain",
    "sobol_normal",
    "sobol_uniform",
    "stationary_distribution",
    "validate_transition_matrix",
    "value_at_risk",
    "weighted_rmse",
]
