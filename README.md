# Stochastic Modelling Lab

[![Tests](https://github.com/afatania09/stochastic-modelling-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/afatania09/stochastic-modelling-lab/actions/workflows/tests.yml)

**A computational laboratory for stochastic processes, SDE simulation, Monte Carlo methods and quantitative finance.**

This repository is designed as a rigorous, reusable and testable stochastic-modelling project rather than a collection of notebooks. The emphasis is on linking mathematical theory to simulation, numerical approximation, convergence analysis and empirical diagnostics.

## Current capabilities

### Stochastic processes
- Standard Brownian motion
- Geometric Brownian motion using exact simulation
- Ornstein-Uhlenbeck mean reversion using exact discretisation
- Cox-Ingersoll-Ross square-root process using full-truncation Euler
- Homogeneous Poisson counting process
- Compound Poisson processes with Gaussian jump sizes
- Merton jump diffusion with drift compensation

### Numerical SDE methods
- Euler-Maruyama
- Milstein
- Pathwise strong-error analysis against exact GBM
- Weak-error analysis against analytical GBM moments
- Log-log convergence-order estimation
- Reproducible seeded simulation

### Monte Carlo methods
- Plain Monte Carlo estimation with standard errors
- Antithetic normal variates
- Optimal linear control variates
- Running mean and standard-error convergence diagnostics

### Validation philosophy

Simulation output is checked against known theoretical properties wherever possible. The test suite verifies, among other things:

- Brownian terminal mean and variance
- GBM expected terminal value and positivity
- OU convergence toward its long-run mean
- CIR non-negativity under full truncation
- Poisson terminal mean against lambda x time
- compound-Poisson theoretical mean
- Merton jump-diffusion expected growth under the compensator
- Milstein's stronger pathwise convergence behaviour relative to Euler on GBM
- control-variate variance reduction
- reproducibility of numerical SDE schemes

The aim is therefore not just to produce simulated paths, but to demonstrate whether an implementation behaves consistently with its mathematical specification.

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

The Euler-Maruyama approximation to a scalar SDE

```text
dX(t) = a(t,X)dt + b(t,X)dW(t)
```

is

```text
X[n+1] = X[n] + a(t[n],X[n]) dt + b(t[n],X[n]) dW[n].
```

Milstein adds the first diffusion-derivative correction. For GBM this raises the expected strong order from approximately 1/2 to approximately 1 under regularity conditions.

For Merton jump diffusion,

```text
dS(t)/S(t-) = (mu - lambda*kappa)dt + sigma dW(t) + (J-1)dN(t),
```

where `N(t)` is Poisson and `kappa = E[J-1]` compensates the jump drift.

## Installation

```bash
git clone https://github.com/afatania09/stochastic-modelling-lab.git
cd stochastic-modelling-lab
python -m venv .venv
pip install -e ".[dev]"
pytest
```

## Examples

Simulate GBM:

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
```

Estimate numerical strong convergence:

```python
from stochastic_lab import gbm_strong_errors

h, errors, order = gbm_strong_errors(
    "milstein",
    step_counts=[8, 16, 32, 64, 128],
    paths=20_000,
)
print(order)
```

Generate a jump-diffusion ensemble:

```python
from stochastic_lab import merton_jump_diffusion

_, paths = merton_jump_diffusion(
    s0=100,
    mu=0.05,
    sigma=0.20,
    jump_intensity=0.8,
    jump_mean=-0.08,
    jump_std=0.18,
    horizon=1,
    steps=252,
    paths=10_000,
    seed=7,
)
```

## Repository structure

```text
src/stochastic_lab/
    processes.py          diffusion and counting processes
    schemes.py            Euler-Maruyama and Milstein
    convergence.py        strong/weak error and convergence diagnostics
    jumps.py              compound Poisson and Merton jump diffusion
    monte_carlo.py        estimators and variance reduction
tests/                    statistical and numerical verification
.github/workflows/        automated CI across supported Python versions
```

## Development roadmap

1. **Foundations** — Brownian motion, GBM, OU, CIR and Poisson processes. ✅
2. **Numerical SDE analysis** — Euler-Maruyama, Milstein, strong and weak convergence. ✅
3. **Jump processes** — compound Poisson and Merton jump diffusion. ✅
4. **Variance reduction** — antithetic variates and control variates. ✅ Core implemented
5. **Importance sampling** — rare-event estimators and likelihood-ratio diagnostics.
6. **Quasi-Monte Carlo** — Sobol sequences and convergence comparison.
7. **Parameter estimation** — MLE, moment estimators and calibration diagnostics.
8. **Stochastic volatility** — Heston simulation and discretisation comparisons.
9. **Regime switching** — Markov switching processes and hidden-state inference.
10. **Applications** — derivative pricing, risk simulation and term-structure modelling.
11. **Research experiments** — reproducible comparisons of simulation bias, convergence and model behaviour.

## Design principles

- mathematical transparency over black-box output;
- deterministic reproducibility through explicit random seeds;
- reusable library code rather than notebook-only implementations;
- theory-versus-simulation validation;
- convergence analysis rather than visual inspection alone;
- tests for numerical and statistical behaviour;
- documented approximations and limitations.

## Disclaimer

This repository is an independent educational and research project. It is not financial or investment advice and its models should not be used for production decisions without appropriate validation, governance and controls.
