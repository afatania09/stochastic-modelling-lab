# Stochastic Modelling Lab

[![Tests](https://github.com/afatania09/stochastic-modelling-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/afatania09/stochastic-modelling-lab/actions/workflows/tests.yml)

**A computational laboratory for stochastic processes, SDE simulation, Monte Carlo methods and quantitative finance.**

This repository is designed as a rigorous, reusable and testable stochastic-modelling project rather than a collection of notebooks. The emphasis is on linking mathematical theory to simulation, numerical approximation and empirical diagnostics.

## Current capabilities

### Stochastic processes
- Standard Brownian motion
- Geometric Brownian motion using exact simulation
- Ornstein-Uhlenbeck mean reversion using exact discretisation
- Cox-Ingersoll-Ross square-root process using full-truncation Euler
- Homogeneous Poisson counting process

### Numerical SDE methods
- Euler-Maruyama
- Milstein
- Reproducible seeded simulation
- Vectorised multi-path simulation where appropriate

### Validation philosophy

Simulation output should be checked against known theoretical properties wherever possible. The test suite currently verifies:

- Brownian terminal mean and variance
- GBM expected terminal value and positivity
- OU convergence toward its long-run mean
- CIR non-negativity under full truncation
- Poisson terminal mean against lambda x time
- reproducibility of numerical SDE schemes

This allows the repository to demonstrate not only how a process is simulated, but whether the implementation behaves consistently with its mathematical specification.

## Mathematical examples

For standard Brownian motion,

```text
W(t) - W(s) ~ Normal(0, t-s)
```

For geometric Brownian motion,

```text
dS(t) = mu S(t) dt + sigma S(t) dW(t)
```

with exact solution

```text
S(t) = S(0) exp[(mu - 0.5 sigma^2)t + sigma W(t)].
```

The OU process is represented by

```text
dX(t) = kappa(theta - X(t))dt + sigma dW(t),
```

while the CIR process uses

```text
dX(t) = kappa(theta - X(t))dt + sigma sqrt(X(t)) dW(t).
```

## Installation

```bash
git clone https://github.com/afatania09/stochastic-modelling-lab.git
cd stochastic-modelling-lab
python -m venv .venv
pip install -e ".[dev]"
pytest
```

## Example

```python
from stochastic_lab import geometric_brownian_motion

_, paths = geometric_brownian_motion(
    s0=100.0,
    mu=0.05,
    sigma=0.20,
    horizon=1.0,
    steps=252,
    paths=10_000,
    seed=42,
)

terminal_values = paths[:, -1]
print(terminal_values.mean())
```

## Repository structure

```text
src/stochastic_lab/       reusable stochastic-process and SDE engine
tests/                    statistical and numerical verification
.github/workflows/        automated CI across supported Python versions
```

## Development roadmap

The project will progress in deliberate stages:

1. **Foundations** — Brownian motion, GBM, OU, CIR and Poisson processes.
2. **Numerical SDE analysis** — Euler-Maruyama, Milstein, strong and weak convergence.
3. **Jump processes** — compound Poisson and Merton jump diffusion.
4. **Variance reduction** — antithetic variates, control variates and importance sampling.
5. **Quasi-Monte Carlo** — Sobol sequences and convergence comparison.
6. **Parameter estimation** — MLE, moment estimators and calibration diagnostics.
7. **Stochastic volatility** — Heston simulation and discretisation comparisons.
8. **Regime switching** — Markov switching processes and hidden-state inference.
9. **Applications** — derivative pricing, risk simulation and term-structure modelling.
10. **Research experiments** — reproducible comparisons of simulation bias, convergence and model behaviour.

## Design principles

- mathematical transparency over black-box output;
- deterministic reproducibility through explicit random seeds;
- reusable library code rather than notebook-only implementations;
- theory-versus-simulation validation;
- tests for numerical and statistical behaviour;
- documented approximations and limitations.

## Disclaimer

This repository is an independent educational and research project. It is not financial or investment advice and its models should not be used for production decisions without appropriate validation, governance and controls.
