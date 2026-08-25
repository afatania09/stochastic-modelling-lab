# Stochastic Modelling Lab

[![Tests](https://github.com/afatania09/stochastic-modelling-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/afatania09/stochastic-modelling-lab/actions/workflows/tests.yml)

**A computational laboratory for stochastic processes, SDE simulation, Monte Carlo methods, parameter estimation and quantitative finance.**

This repository is designed as a rigorous, reusable and testable stochastic-modelling project rather than a collection of notebooks. The emphasis is on linking mathematical theory to simulation, numerical approximation, convergence analysis, estimation and empirical diagnostics.

## Current capabilities

### Stochastic processes
- Standard Brownian motion
- Geometric Brownian motion using exact simulation
- Ornstein-Uhlenbeck mean reversion using exact discretisation
- Cox-Ingersoll-Ross square-root process using full-truncation Euler
- Homogeneous Poisson counting process
- Compound Poisson processes with Gaussian jump sizes
- Merton jump diffusion with drift compensation
- Heston stochastic volatility with correlated shocks
- Finite-state Markov regime switching

### Numerical SDE methods
- Euler-Maruyama
- Milstein
- Pathwise strong-error analysis against exact GBM
- Weak-error analysis against analytical GBM moments
- Log-log convergence-order estimation
- Full-truncation treatment for square-root variance processes
- Reproducible seeded simulation

### Monte Carlo methods
- Plain Monte Carlo estimation with standard errors
- Antithetic normal variates
- Optimal linear control variates
- Running mean and standard-error convergence diagnostics
- Importance sampling for rare Gaussian and lognormal tail events
- Sobol low-discrepancy sequences
- Randomised quasi-Monte Carlo integration with replication-based error estimates

### Parameter estimation and calibration
- GBM maximum-likelihood estimation from log returns
- OU estimation using its exact AR(1) transition representation
- Parameter recovery diagnostics
- Weighted RMSE calibration loss
- Relative RMSE
- Parameter-bound validation

### Stochastic volatility
The Heston simulator implements

```text
dS(t) = mu S(t) dt + sqrt(v(t)) S(t) dW1(t)
dv(t) = kappa(theta-v(t))dt + xi sqrt(v(t)) dW2(t)
corr(dW1,dW2) = rho
```

using full-truncation Euler for the variance process and log-Euler evolution for prices. The package also exposes the Feller margin

```text
2*kappa*theta - xi^2
```

so parameter sets can be checked explicitly rather than treated as black boxes.

### Regime switching
Finite-state Markov chains can be simulated from a row-stochastic transition matrix. The package computes stationary distributions and supports Gaussian returns whose conditional mean and volatility depend on the latent regime. This makes it possible to model calm/stress states and study persistence, long-run regime occupancy and volatility mixtures.

## Validation philosophy

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
- importance-sampling agreement with the analytical Gaussian tail
- material standard-error reduction for a rare-event estimator
- Sobol point bounds and transformed-normal moments
- quasi-Monte Carlo integration against an analytical polynomial integral
- GBM parameter recovery from simulated data
- OU parameter-estimation sanity checks
- Heston price positivity and non-negative variance paths
- Feller-condition diagnostics
- Markov-chain stationary-distribution consistency
- long-run regime frequencies versus stationary probabilities
- regime-dependent volatility separation
- calibration-loss behaviour
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

For a rare Gaussian event `P[Z > a]`, importance sampling draws under a shifted law and applies the likelihood ratio

```text
L(x) = exp(-theta*x + 0.5*theta^2).
```

For quasi-Monte Carlo, low-discrepancy Sobol points replace independent uniforms; randomised scrambles permit repeated estimates and a practical standard-error calculation.

For an OU process sampled every `dt`, the exact transition can be written as an AR(1) model with

```text
phi = exp(-kappa*dt),
```

which permits transparent estimation of mean reversion, long-run mean and diffusion scale.

## Installation

```bash
git clone https://github.com/afatania09/stochastic-modelling-lab.git
cd stochastic-modelling-lab
python -m venv .venv
pip install -e ".[dev]"
pytest
```

## Examples

### Simulate GBM

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

### Estimate numerical strong convergence

```python
from stochastic_lab import gbm_strong_errors

h, errors, order = gbm_strong_errors(
    "milstein",
    step_counts=[8, 16, 32, 64, 128],
    paths=20_000,
)
print(order)
```

### Estimate a rare 4-sigma event

```python
from stochastic_lab import normal_tail_probability_importance_sampling

estimate, se = normal_tail_probability_importance_sampling(
    threshold=4.0,
    simulations=200_000,
    seed=1,
)
print(estimate, se)
```

### Randomised quasi-Monte Carlo integration

```python
from stochastic_lab import qmc_integrate

estimate, se = qmc_integrate(
    lambda u: u[:, 0] ** 2 + u[:, 1] ** 2,
    dimension=2,
    power=10,
    replications=8,
    seed=123,
)
```

### Simulate Heston stochastic volatility

```python
from stochastic_lab import heston_paths

_, prices, variances = heston_paths(
    s0=100.0,
    v0=0.04,
    mu=0.05,
    kappa=2.0,
    theta=0.04,
    xi=0.30,
    rho=-0.7,
    horizon=1.0,
    steps=252,
    paths=10_000,
    seed=7,
)
```

### Model calm and stress regimes

```python
import numpy as np
from stochastic_lab import markov_switching_returns, stationary_distribution

transition = np.array([[0.97, 0.03], [0.10, 0.90]])
print(stationary_distribution(transition))

states, returns = markov_switching_returns(
    transition,
    means=np.array([0.0004, -0.0008]),
    volatilities=np.array([0.008, 0.03]),
    steps=5_000,
    seed=42,
)
```

A runnable end-to-end demonstration is available at `experiments/advanced_models_demo.py`.

## Repository structure

```text
src/stochastic_lab/
    processes.py          diffusion and counting processes
    schemes.py            Euler-Maruyama and Milstein
    convergence.py        strong/weak error and convergence diagnostics
    jumps.py              compound Poisson and Merton jump diffusion
    monte_carlo.py        estimators and variance reduction
    rare_events.py        importance sampling and tail-event estimation
    qmc.py                Sobol low-discrepancy and randomised QMC tools
    estimation.py         GBM and OU parameter estimation
    calibration.py        fit metrics and parameter diagnostics
    heston.py             stochastic-volatility simulation
    regime_switching.py   latent Markov regimes and switching returns
experiments/              reproducible research-style demonstrations
tests/                    statistical and numerical verification
.github/workflows/        automated CI across supported Python versions
```

## Development roadmap

1. **Foundations** — Brownian motion, GBM, OU, CIR and Poisson processes. ✅
2. **Numerical SDE analysis** — Euler-Maruyama, Milstein, strong and weak convergence. ✅
3. **Jump processes** — compound Poisson and Merton jump diffusion. ✅
4. **Variance reduction** — antithetic variates and control variates. ✅
5. **Importance sampling** — rare-event estimators and likelihood-ratio diagnostics. ✅
6. **Quasi-Monte Carlo** — Sobol sequences and randomised QMC integration. ✅
7. **Parameter estimation** — GBM MLE, OU transition estimation and calibration diagnostics. ✅ Core implemented
8. **Stochastic volatility** — Heston simulation, Feller diagnostics and variance-path analysis. ✅ Core implemented
9. **Regime switching** — Markov switching processes, stationary distributions and state-dependent returns. ✅ Core implemented
10. **Applications** — derivative pricing, risk simulation and term-structure modelling. Next
11. **Research experiments** — reproducible comparisons of simulation bias, convergence, estimation error and model behaviour. In progress

## Design principles

- mathematical transparency over black-box output;
- deterministic reproducibility through explicit random seeds;
- reusable library code rather than notebook-only implementations;
- theory-versus-simulation validation;
- convergence analysis rather than visual inspection alone;
- explicit estimation and calibration diagnostics;
- tests for numerical and statistical behaviour;
- documented approximations and limitations.

## Disclaimer

This repository is an independent educational and research project. It is not financial or investment advice and its models should not be used for production decisions without appropriate validation, governance and controls.
