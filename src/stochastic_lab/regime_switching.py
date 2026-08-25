"""Finite-state Markov regime-switching models."""

from __future__ import annotations

import numpy as np


def validate_transition_matrix(transition: np.ndarray) -> np.ndarray:
    """Validate and return a row-stochastic transition matrix."""
    matrix = np.asarray(transition, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("transition must be square")
    if np.any(matrix < 0):
        raise ValueError("transition probabilities must be non-negative")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("each transition row must sum to one")
    return matrix


def stationary_distribution(transition: np.ndarray) -> np.ndarray:
    """Compute the stationary distribution of an ergodic transition matrix."""
    matrix = validate_transition_matrix(transition)
    eigenvalues, eigenvectors = np.linalg.eig(matrix.T)
    idx = int(np.argmin(np.abs(eigenvalues - 1.0)))
    vector = np.real(eigenvectors[:, idx])
    vector = np.maximum(vector, 0.0)
    if vector.sum() == 0:
        vector = np.abs(np.real(eigenvectors[:, idx]))
    return vector / vector.sum()


def simulate_markov_chain(
    transition: np.ndarray,
    steps: int,
    paths: int = 1,
    initial_state: int = 0,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate discrete-state Markov chains."""
    matrix = validate_transition_matrix(transition)
    states_n = matrix.shape[0]
    if steps <= 0 or paths <= 0:
        raise ValueError("steps and paths must be positive")
    if initial_state < 0 or initial_state >= states_n:
        raise ValueError("initial_state is invalid")

    rng = np.random.default_rng(seed)
    states = np.empty((paths, steps + 1), dtype=int)
    states[:, 0] = initial_state
    for t in range(steps):
        u = rng.random(paths)
        for p in range(paths):
            probs = np.cumsum(matrix[states[p, t]])
            states[p, t + 1] = int(np.searchsorted(probs, u[p], side="right"))
    return states


def markov_switching_returns(
    transition: np.ndarray,
    means: np.ndarray,
    volatilities: np.ndarray,
    steps: int,
    paths: int = 1,
    initial_state: int = 0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate Gaussian returns conditional on latent Markov regimes."""
    matrix = validate_transition_matrix(transition)
    means = np.asarray(means, dtype=float)
    volatilities = np.asarray(volatilities, dtype=float)
    if means.shape != (matrix.shape[0],) or volatilities.shape != means.shape:
        raise ValueError("means and volatilities must match number of regimes")
    if np.any(volatilities <= 0):
        raise ValueError("volatilities must be positive")

    states = simulate_markov_chain(matrix, steps, paths, initial_state, seed)
    rng = np.random.default_rng(None if seed is None else seed + 1)
    z = rng.standard_normal((paths, steps))
    returns = means[states[:, 1:]] + volatilities[states[:, 1:]] * z
    return states, returns
