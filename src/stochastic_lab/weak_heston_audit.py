"""Quadrature-refinement audit for the manufactured 2D weak Heston PDE.

Run with: python -m stochastic_lab.weak_heston_audit
No neural-network training or pilot results are generated.
"""
from __future__ import annotations

import json

import numpy as np

from .heston_operator import heston_backward_residual
from .weak_heston_2d import weak_heston_2d


def run_audit(orders: tuple[int, ...] = (8, 12, 24, 36, 48, 64, 80)) -> dict:
    """Compare weak and strong integrals on a smooth non-polynomial field."""
    params = {"rate": 0.03, "kappa": 1.7, "theta": 0.05, "xi": 0.4, "rho": -0.5}

    def fields(s, v):
        # V=exp(S/100)*(1+v**2), with analytic derivatives.
        e = np.exp(s / 100)
        return {
            "value": e * (1 + v**2),
            "dt": np.zeros_like(s),
            "ds": e * (1 + v**2) / 100,
            "dv": 2 * v * e,
        }

    def phi(s, v):
        return np.exp(s / 120) * (1 + v)

    def phi_s(s, v):
        return phi(s, v) / 120

    def phi_v(s, v):
        return np.exp(s / 120)

    from numpy.polynomial.legendre import leggauss

    rows = []
    for order in orders:
        weak = weak_heston_2d(
            spot_bounds=(80, 120), variance_bounds=(0, 0.2),
            order=order, fields=fields, test=phi, test_ds=phi_s,
            test_dv=phi_v, **params
        )
        nodes, weights = leggauss(order)
        s, v = np.meshgrid(100 + 20 * nodes, 0.1 + 0.1 * nodes, indexing="ij")
        f = fields(s, v)
        e = np.exp(s / 100)
        strong = heston_backward_residual(
            spot=s, variance=v, value=f["value"], dt=f["dt"],
            ds=f["ds"], dv=f["dv"], dss=f["value"] / 10000,
            dsv=2 * v * e / 100, dvv=2 * e, **params
        )
        direct = float(np.sum(weights[:, None] * weights[None, :]
                              * phi(s, v) * strong) * 2.0)
        rows.append({
            "order": order, "weak": float(weak), "strong": direct,
            "absolute_difference": abs(float(weak) - direct),
        })
    return {"experiment": "manufactured_2d_quadrature_consistency",
            "orders": list(orders), "rows": rows,
            "note": "Analytic manufactured field; not a PINN result."}


if __name__ == "__main__":
    print(json.dumps(run_audit(), indent=2))
