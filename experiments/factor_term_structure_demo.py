"""Reproducible demonstration of factor, multivariate and term-structure tools."""

from __future__ import annotations

import numpy as np

from stochastic_lab import (
    correlated_brownian_motion,
    covariance_explained,
    pca_factor_decomposition,
    vasicek_yield_curve,
)


def main() -> None:
    correlation = np.array(
        [
            [1.0, 0.70, 0.35],
            [0.70, 1.0, 0.50],
            [0.35, 0.50, 1.0],
        ]
    )
    _, brownian = correlated_brownian_motion(
        correlation, horizon=1.0, steps=252, paths=20_000, seed=42
    )
    terminal = brownian[:, :, -1]
    print("Target correlation:\n", correlation)
    print("Empirical terminal correlation:\n", np.corrcoef(terminal.T))

    rng = np.random.default_rng(11)
    latent = rng.normal(size=(5_000, 2))
    loadings = np.array([[1.0, 0.2], [0.8, -0.3], [0.6, 0.5], [0.4, -0.6]])
    panel = latent @ loadings.T + 0.05 * rng.normal(size=(5_000, 4))
    fit = pca_factor_decomposition(panel, n_components=2)
    explained = covariance_explained(fit["eigenvalues"], 2)
    print(f"Variance explained by two factors: {explained:.4f}")

    maturities = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
    yields = vasicek_yield_curve(
        short_rate=0.03,
        maturities=maturities,
        kappa=1.2,
        theta=0.045,
        sigma=0.012,
    )
    print("Vasicek maturities:", maturities)
    print("Vasicek zero-coupon yields:", yields)


if __name__ == "__main__":
    main()
