"""Smoke-test reproducible weak Heston quadrature audit."""
import pytest

from stochastic_lab.weak_heston_audit import run_audit


def test_manufactured_audit_consistency():
    report = run_audit((8, 12, 24))
    assert report["orders"] == [8, 12, 24]
    assert len(report["rows"]) == 3
    for row in report["rows"]:
        assert row["weak"] == pytest.approx(row["strong"], rel=1e-10, abs=1e-8)
