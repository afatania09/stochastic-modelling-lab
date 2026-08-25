"""Principal-component factor decomposition and covariance reconstruction."""

from __future__ import annotations

import numpy as np


def pca_factor_decomposition(
    data: np.ndarray,
    n_components: int | None = None,
    standardise: bool = False,
) -> dict[str, np.ndarray]:
    """Fit a PCA factor model using an eigen-decomposition of the sample covariance."""
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("data must be a two-dimensional array with at least two observations")
    mean = x.mean(axis=0)
    centred = x - mean
    scale = np.ones(x.shape[1])
    if standardise:
        scale = centred.std(axis=0, ddof=1)
        if np.any(scale <= 0):
            raise ValueError("cannot standardise zero-variance columns")
        centred = centred / scale
    cov = np.cov(centred, rowvar=False, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    eigenvectors = eigenvectors[:, order]
    if n_components is None:
        n_components = x.shape[1]
    if not 1 <= n_components <= x.shape[1]:
        raise ValueError("n_components must lie between one and the number of variables")
    components = eigenvectors[:, :n_components]
    scores = centred @ components
    explained_ratio = eigenvalues / eigenvalues.sum() if eigenvalues.sum() > 0 else eigenvalues
    return {
        "mean": mean,
        "scale": scale,
        "components": components,
        "scores": scores,
        "eigenvalues": eigenvalues,
        "explained_variance_ratio": explained_ratio,
    }


def reconstruct_from_factors(
    scores: np.ndarray,
    components: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct observations from PCA scores and loadings."""
    z = np.asarray(scores, dtype=float) @ np.asarray(components, dtype=float).T
    if scale is not None:
        z = z * np.asarray(scale, dtype=float)
    return z + np.asarray(mean, dtype=float)


def covariance_explained(eigenvalues: np.ndarray, n_components: int) -> float:
    """Fraction of total covariance variation explained by the leading factors."""
    values = np.asarray(eigenvalues, dtype=float)
    if values.ndim != 1 or np.any(values < 0) or not 1 <= n_components <= len(values):
        raise ValueError("invalid eigenvalues or n_components")
    total = float(values.sum())
    return float(values[:n_components].sum() / total) if total > 0 else 0.0
