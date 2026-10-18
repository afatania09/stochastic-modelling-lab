"""Manufactured quadrature verification; no neural-network claims."""
import numpy as np
import pytest

from stochastic_lab.weak_quadrature import gauss_legendre_integral, refinement_table


def test_polynomial_exactness():
    # An n-point Gauss-Legendre rule integrates polynomials up to degree 2n-1.
    for order in (2, 4, 12):
        result = gauss_legendre_integral(lambda x, exponent=2 * order - 2: x ** exponent, -1, 1, order)
        assert result == pytest.approx(2 / (2 * order - 1), abs=1e-12)


def test_smooth_nonpolynomial_refinement():
    rows = refinement_table(np.exp, 0, 1)
    assert abs(rows[-1]["estimate"] - np.expm1(1)) < 1e-12


def test_reject_invalid_input():
    with pytest.raises(ValueError):
        gauss_legendre_integral(np.exp, 1, 0, 12)
    with pytest.raises(ValueError):
        gauss_legendre_integral(np.exp, 0, 1, 0)
    with pytest.raises(ValueError):
        gauss_legendre_integral(lambda x: np.array([1.0]), 0, 1, 12)
    with pytest.raises(ValueError):
        refinement_table(np.exp, 0, 1, (24, 12))
