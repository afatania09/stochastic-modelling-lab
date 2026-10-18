"""Two-dimensional spatial weak Heston residual in divergence form.

Integrates the backward Heston PDE over a rectangular (spot, variance)
domain at a fixed time. The full diffusion tensor is integrated by parts
once, including all four boundary fluxes. No neural network is required.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.polynomial.legendre import leggauss


def weak_heston_2d(
    *,
    spot_bounds: tuple[float, float],
    variance_bounds: tuple[float, float],
    rate: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    order: int,
    fields: Callable[[np.ndarray, np.ndarray], dict[str, np.ndarray]],
    test: Callable[[np.ndarray, np.ndarray], np.ndarray],
    test_ds: Callable[[np.ndarray, np.ndarray], np.ndarray],
    test_dv: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> float:
    """Compute integral phi*(V_t+L_H V-rV) dS dv using first derivatives.

    fields returns value, dt, ds, dv arrays. Boundary traces must exist.
    Variance v=0 is allowed; no Feller condition is imposed.
    """
    a, b = spot_bounds
    c, d = variance_bounds
    if not np.isfinite([a, b, c, d, rate, kappa, theta, xi, rho]).all():
        raise ValueError("all parameters must be finite")
    if not (0 < a < b and 0 <= c < d):
        raise ValueError("invalid integration bounds")
    if kappa <= 0 or theta <= 0 or xi <= 0 or abs(rho) > 1:
        raise ValueError("invalid Heston parameters")
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError("order must be a positive integer")

    nodes, weights = leggauss(order)
    sn = (b - a) * nodes / 2 + (a + b) / 2
    vn = (d - c) * nodes / 2 + (c + d) / 2
    sw = weights * (b - a) / 2
    vw = weights * (d - c) / 2

    def checked(values, shape):
        arr = np.asarray(values, dtype=float)
        if arr.shape != shape or not np.isfinite(arr).all():
            raise ValueError("fields and test functions must match evaluation shape")
        return arr

    def evaluate(s, v):
        shape = s.shape
        f = fields(s, v)
        if any(key not in f for key in ("value", "dt", "ds", "dv")):
            raise ValueError("missing fields")
        val, dt, ds, dv = (checked(f[key], shape)
                            for key in ("value", "dt", "ds", "dv"))
        phi = checked(test(s, v), shape)
        ps = checked(test_ds(s, v), shape)
        pv = checked(test_dv(s, v), shape)
        # Symmetric diffusion tensor D:
        # D11=0.5*v*S**2, D12=0.5*rho*xi*v*S, D22=0.5*xi**2*v.
        d11 = 0.5 * v * s**2
        d12 = 0.5 * rho * xi * v * s
        d22 = 0.5 * xi**2 * v
        flux_s = d11 * ds + d12 * dv
        flux_v = d12 * ds + d22 * dv
        # div(D) components: partial_S D11 + partial_v D12,
        # partial_S D12 + partial_v D22.
        div_s = v * s + 0.5 * rho * xi * s
        div_v = 0.5 * rho * xi * v + 0.5 * xi**2
        drift_s = rate * s
        drift_v = kappa * (theta - v)
        volume = (
            phi * (dt - rate * val
                   + (drift_s - div_s) * ds
                   + (drift_v - div_v) * dv)
            - ps * flux_s - pv * flux_v
        )
        return volume, phi * flux_s, phi * flux_v

    s, v = np.meshgrid(sn, vn, indexing="ij")
    volume, _, _ = evaluate(s, v)
    result = float(np.sum(volume * sw[:, None] * vw[None, :]))

    # Flux through S=b minus flux through S=a.
    for boundary, sign in ((b, 1), (a, -1)):
        ss = np.full_like(vn, boundary)
        _, flux_s, _ = evaluate(ss, vn)
        result += sign * float(np.dot(vw, flux_s))

    # Flux through v=d minus flux through v=c.
    for boundary, sign in ((d, 1), (c, -1)):
        vv = np.full_like(sn, boundary)
        _, _, flux_v = evaluate(sn, vv)
        result += sign * float(np.dot(sw, flux_v))
    return result
