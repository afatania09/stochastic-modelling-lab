"""Independent Gaussian quadrature diagnostics for weak-form experiments.

This module does not train a PINN. It verifies numerical integration on
manufactured functions before weak residuals are used in training.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.polynomial.legendre import leggauss


def gauss_legendre_integral(
    function: Callable[[np.ndarray], np.ndarray],
    lower: float,
    upper: float,
    order: int,
) -> float:
    """Integrate a vectorised scalar function on a finite interval."""
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("Require finite bounds with upper > lower")
    if not isinstance(order, (int, np.integer)) or isinstance(order, (bool, np.bool_)) or order < 1:
        raise ValueError("order must be a positive integer")
    nodes, weights = leggauss(int(order))
    x = 0.5 * (upper - lower) * nodes + 0.5 * (upper + lower)
    values = np.asarray(function(x), dtype=float)
    if values.shape != x.shape or not np.all(np.isfinite(values)):
        raise ValueError("function must return finite values matching the node shape")
    return float(0.5 * (upper - lower) * np.dot(weights, values))


def refinement_table(
    function: Callable[[np.ndarray], np.ndarray],
    lower: float,
    upper: float,
    orders: tuple[int, ...] = (12, 24, 36, 48, 64, 80),
) -> list[dict[str, float | int | None]]:
    """Return quadrature estimates and successive absolute changes.

    Successive agreement is a diagnostic, not proof of convergence.
    """
    if not orders or any(order < 1 for order in orders) or any(
        b <= a for a, b in zip(orders, orders[1:])
    ):
        raise ValueError("orders must be strictly increasing positive integers")
    rows: list[dict[str, float | int | None]] = []
    previous: float | None = None
    for order in orders:
        value = gauss_legendre_integral(function, lower, upper, order)
        rows.append(
            {"order": order, "estimate": value,
             "absolute_change": None if previous is None else abs(value - previous)}
        )
        previous = value
    return rows
