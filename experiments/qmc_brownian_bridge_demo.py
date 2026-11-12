"""Compare pseudo-random MC, forward Sobol QMC and Brownian-bridge Sobol QMC.

The payoff is a discretely monitored arithmetic-average Asian call under GBM.
The experiment reports price estimates across randomized replications so the
variance effect of path construction can be inspected directly.
"""

from __future__ import annotations

import numpy as np

from stochastic_lab import sobol_brownian_bridge, sobol_normal


def asian_call_from_brownian(
    brownian: np.ndarray,
    s0: float = 100.0,
    strike: float = 100.0,
    rate: float = 0.03,
    sigma: float = 0.20,
    horizon: float = 1.0,
) -> float:
    """Price a discrete arithmetic-average Asian call from Brownian paths."""
    steps = brownian.shape[1] - 1
    time = np.linspace(0.0, horizon, steps + 1)
    prices = s0 * np.exp(
        (rate - 0.5 * sigma**2) * time[None, :] + sigma * brownian
    )
    average = prices[:, 1:].mean(axis=1)
    payoff = np.maximum(average - strike, 0.0)
    return float(np.exp(-rate * horizon) * payoff.mean())


def pseudo_random_estimate(
    paths: int,
    steps: int,
    seed: int,
    horizon: float = 1.0,
) -> float:
    """Asian-call estimate from ordinary pseudo-random Brownian increments."""
    rng = np.random.default_rng(seed)
    dt = horizon / steps
    increments = np.sqrt(dt) * rng.standard_normal((paths, steps))
    brownian = np.column_stack([np.zeros(paths), np.cumsum(increments, axis=1)])
    return asian_call_from_brownian(brownian, horizon=horizon)


def forward_sobol_estimate(
    power: int,
    steps: int,
    seed: int,
    horizon: float = 1.0,
) -> float:
    """Asian-call estimate using Sobol normals in chronological increment order."""
    normals = sobol_normal(steps, power, scramble=True, seed=seed)
    dt = horizon / steps
    increments = np.sqrt(dt) * normals
    brownian = np.column_stack(
        [np.zeros(normals.shape[0]), np.cumsum(increments, axis=1)]
    )
    return asian_call_from_brownian(brownian, horizon=horizon)


def bridge_sobol_estimate(
    power: int,
    steps: int,
    seed: int,
    horizon: float = 1.0,
) -> float:
    """Asian-call estimate using Sobol normals with Brownian-bridge ordering."""
    _, brownian = sobol_brownian_bridge(
        steps=steps,
        power=power,
        horizon=horizon,
        scramble=True,
        seed=seed,
    )
    return asian_call_from_brownian(brownian, horizon=horizon)


def summarize(name: str, estimates: np.ndarray) -> None:
    """Print the across-replication mean and standard deviation."""
    print(
        f"{name:>22}: mean={estimates.mean():.6f} "
        f"replication_sd={estimates.std(ddof=1):.6f}"
    )


def main() -> None:
    steps = 32
    power = 12
    replications = 16
    paths = 2**power
    seeds = np.arange(100, 100 + replications)

    pseudo = np.array(
        [pseudo_random_estimate(paths, steps, int(seed)) for seed in seeds]
    )
    forward = np.array(
        [forward_sobol_estimate(power, steps, int(seed)) for seed in seeds]
    )
    bridge = np.array(
        [bridge_sobol_estimate(power, steps, int(seed)) for seed in seeds]
    )

    print(f"Asian call experiment: {paths} paths x {steps} monitoring steps")
    summarize("pseudo-random MC", pseudo)
    summarize("forward Sobol QMC", forward)
    summarize("bridge Sobol QMC", bridge)


if __name__ == "__main__":
    main()
