"""Semi-analytical European call pricing under the risk-neutral Heston model.

Uses the characteristic function and a damped Fourier integral (Lewis form).
The integration is numerical; this is not a closed-form elementary solution.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad


def heston_characteristic_function(
    u: complex | np.ndarray,
    s0: float,
    rate: float,
    maturity: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
) -> complex | np.ndarray:
    """Characteristic function of log(S_T), under risk-neutral Heston dynamics."""
    if s0 <= 0 or maturity < 0 or v0 < 0 or kappa <= 0 or theta < 0:
        raise ValueError("invalid spot, maturity, variance or mean reversion")
    if xi <= 0 or not -1 <= rho <= 1:
        raise ValueError("xi must be positive and rho must lie in [-1,1]")
    u = np.asarray(u, dtype=complex)
    if maturity == 0:
        return np.exp(1j * u * np.log(s0))
    b = kappa - rho * xi * 1j * u
    d = np.sqrt(b * b + xi * xi * (u * u + 1j * u))
    # Stable 'little Heston trap' form for ordinary admissible parameters.
    g = (b - d) / (b + d)
    e = np.exp(-d * maturity)
    log_ratio = np.log((1 - g * e) / (1 - g))
    c = (
        1j * u * (np.log(s0) + rate * maturity)
        + kappa * theta / (xi * xi)
        * ((b - d) * maturity - 2 * log_ratio)
    )
    d_term = (b - d) / (xi * xi) * ((1 - e) / (1 - g * e))
    result = np.exp(c + d_term * v0)
    return result.item() if result.ndim == 0 else result


def heston_european_call_fourier(
    s0: float,
    strike: float,
    rate: float,
    maturity: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    *,
    integration_limit: float = 150.0,
    epsabs: float = 1e-8,
) -> float:
    """Price a call by the Lewis damped characteristic-function integral.

    Assumes no dividends. Numerical quadrature is truncated at integration_limit;
    convergence must be checked by increasing the truncation limit.
    """
    if strike <= 0 or integration_limit <= 0 or epsabs <= 0:
        raise ValueError("strike, integration_limit and epsabs must be positive")
    if maturity < 0:
        raise ValueError("maturity must be non-negative")
    if maturity == 0:
        return max(s0 - strike, 0.0)

    log_strike = np.log(strike)

    def integrand(x: float) -> float:
        z = x - 0.5j
        phi = heston_characteristic_function(
            z, s0, rate, maturity, v0, kappa, theta, xi, rho
        )
        return float(np.real(np.exp(-1j * x * log_strike) * phi)
                     / (x * x + 0.25))

    integral, _ = quad(integrand, 0, integration_limit, epsabs=epsabs, limit=500)
    price = s0 - np.exp(-rate * maturity) * np.sqrt(strike) * integral / np.pi
    return float(price)
