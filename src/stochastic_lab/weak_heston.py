"""One-dimensional-in-spot weak residual for the Heston pricing PDE.

The variance coordinate is held fixed while integrating over spot. This is
a partial weak formulation, NOT the full two-dimensional weak Heston PDE.
Only the V_SS term is integrated by parts. The mixed V_Sv and V_vv
derivatives remain in strong form. Test functions need one spot derivative.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.polynomial.legendre import leggauss


def weak_spot_residual(
    *,
    lower: float,
    upper: float,
    variance: float,
    rate: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    order: int,
    fields: Callable[[np.ndarray], dict[str, np.ndarray]],
    test: Callable[[np.ndarray], np.ndarray],
    test_ds: Callable[[np.ndarray], np.ndarray],
) -> float:
    """Integrate phi*(V_t+L V-rV) ds, integrating V_SS by parts.

    fields(s) returns arrays named value, dt, ds, dv, dsv, dvv.
    The boundary flux [0.5*v*s**2*V_s*phi]_(lower)^(upper)
    is included explicitly, so phi need not vanish at the endpoints.
    """
    if not np.isfinite([lower, upper, variance, rate, kappa, theta, xi, rho]).all():
        raise ValueError("parameters must be finite")
    if lower <= 0 or upper <= lower or variance < 0:
        raise ValueError("require 0 < lower < upper and variance >= 0")
    if kappa <= 0 or theta <= 0 or xi <= 0 or abs(rho) > 1:
        raise ValueError("invalid Heston parameters")
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError("order must be a positive integer")

    nodes, weights = leggauss(order)
    spots = (upper - lower) * nodes / 2 + (upper + lower) / 2
    scale = (upper - lower) / 2

    def checked(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        result = np.asarray(arr, dtype=float)
        if result.shape != shape or not np.isfinite(result).all():
            raise ValueError("fields and test functions must match node shapes")
        return result

    f = fields(spots)
    required = ("value", "dt", "ds", "dv", "dsv", "dvv")
    if any(key not in f for key in required):
        raise ValueError("fields missing required derivatives")
    val, dt, ds, dv, dsv, dvv = (
        checked(f[key], spots.shape) for key in required
    )
    phi = checked(test(spots), spots.shape)
    phi_s = checked(test_ds(spots), spots.shape)

    a = 0.5 * variance * spots**2
    a_s = variance * spots
    remaining = (
        dt + rate * spots * ds + kappa * (theta - variance) * dv
        + rho * xi * variance * spots * dsv
        + 0.5 * xi**2 * variance * dvv - rate * val
    )
    integrand = phi * remaining - (a * phi_s + a_s * phi) * ds
    integral = scale * np.dot(weights, integrand)

    endpoints = np.array([lower, upper], dtype=float)
    boundary_ds = checked(fields(endpoints)["ds"], endpoints.shape)
    boundary_phi = checked(test(endpoints), endpoints.shape)
    flux = 0.5 * variance * endpoints**2 * boundary_ds * boundary_phi
    return float(integral + flux[1] - flux[0])
