# Stochastic Modelling Lab

[![Tests](https://github.com/afatania09/stochastic-modelling-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/afatania09/stochastic-modelling-lab/actions/workflows/tests.yml)

**A computational laboratory for stochastic processes, numerical SDEs, Monte Carlo methods, stochastic volatility, factor models, term-structure modelling and quantitative finance.**

This repository is built as a reusable, testable stochastic-modelling library rather than a notebook collection. The core principle is simple: every model should connect mathematical specification, simulation or estimation, and numerical validation.

## Capability map

### Core stochastic processes
- Brownian motion
- correlated Brownian motion
- geometric Brownian motion
- correlated multi-asset GBM
- Ornstein-Uhlenbeck mean reversion
- CIR square-root diffusion
- Poisson and compound-Poisson processes
- Merton jump diffusion
- Heston stochastic volatility
- finite-state Markov regime switching
- Vasicek short-rate dynamics

### Numerical SDE methods
- Euler-Maruyama
- Milstein
- strong and weak convergence diagnostics
- pathwise GBM comparison against exact solutions
- log-log convergence-order estimation
- full-truncation schemes for non-negative variance processes

### Monte Carlo and variance reduction
- plain Monte Carlo estimators with standard errors
- antithetic variates
- optimal linear control variates
- running convergence diagnostics
- importance sampling for rare events
- Sobol low-discrepancy sequences
- randomised quasi-Monte Carlo

### Estimation and calibration
- GBM maximum-likelihood estimation
- OU parameter recovery from exact AR(1) transitions
- weighted and relative RMSE diagnostics
- parameter-bound validation
- Heston price-surface RMSE diagnostics
- common-seed Heston Monte Carlo price surfaces

### Multivariate and factor modelling
- positive-semidefinite correlation validation
- correlated Gaussian shock generation
- correlated multi-asset diffusion simulation
- PCA factor decomposition
- explained-variance diagnostics
- full-rank and reduced-rank reconstruction

### Term-structure modelling
The Vasicek model

```text
dr(t) = kappa(theta-r(t))dt + sigma dW(t)
```

is implemented with exact discretisation. Analytical zero-coupon bond prices are available in affine form

```text
P(0,T) = A(T) exp(-B(T) r0),
```

with continuously compounded zero-coupon yields and complete yield-curve evaluation across arbitrary maturities.

### Stochastic volatility and derivatives
The Heston simulator implements

```text
dS(t) = mu S(t)dt + sqrt(v(t)) S(t)dW1(t)
dv(t) = kappa(theta-v(t))dt + xi sqrt(v(t))dW2(t)
corr(dW1,dW2) = rho
```

with full-truncation variance dynamics. The package now also includes European-call Monte Carlo pricing, strike/maturity Heston price surfaces and calibration-loss diagnostics. In the low vol-of-vol limit, Heston Monte Carlo is explicitly tested against Black-Scholes behaviour.

### Risk applications
- empirical Value at Risk
- Expected Shortfall
- drawdown paths
- maximum drawdown
- regime-dependent return simulation
- option-pricing uncertainty through Monte Carlo standard errors

## Validation philosophy

The repository is designed around theory-versus-computation checks rather than visual output alone. The automated test suite includes checks for:

- Brownian terminal mean and variance
- target correlation recovery in multivariate Brownian motion
- GBM expected value and positivity
- correlated GBM dimensional consistency
- OU mean reversion and parameter-estimation sanity
- CIR non-negativity
- Poisson and compound-Poisson theoretical moments
- Merton jump-diffusion expected growth
- Euler/Milstein convergence behaviour
- importance-sampling agreement with analytical Gaussian tails
- standard-error reduction from variance-reduction techniques
- Sobol point properties and QMC integration accuracy
- Heston variance non-negativity and Feller diagnostics
- Heston pricing consistency with Black-Scholes in an appropriate limiting case
- Markov stationary-distribution consistency and long-run regime frequencies
- PCA reconstruction and covariance-explained diagnostics
- Vasicek exact-discretisation long-run behaviour
- Vasicek analytical bond-price and yield identities

## Mathematical examples

### Brownian motion

```text
W(t)-W(s) ~ Normal(0,t-s)
```

### Geometric Brownian motion

```text
dS(t) = mu S(t)dt + sigma S(t)dW(t)
S(t) = S(0) exp[(mu-0.5 sigma^2)t + sigma W(t)]
```

### Merton jump diffusion

```text
dS(t)/S(t-) = (mu-lambda*kappa_J)dt + sigma dW(t) + (J-1)dN(t)
```

### OU exact transition

```text
phi = exp(-kappa*dt)
X(t+dt) = theta + phi[X(t)-theta] + epsilon
```

### PCA factor representation

```text
X_centered = Scores @ Loadings.T + residual
```

The leading eigenvectors of the sample covariance matrix define orthogonal statistical factors, while the cumulative leading eigenvalues quantify covariance variation explained.

## Installation

```bash
git clone https://github.com/afatania09/stochastic-modelling-lab.git
cd stochastic-modelling-lab
python -m venv .venv
pip install -e ".[dev]"
pytest
```

## Examples

### Strong SDE convergence

```python
from stochastic_lab import gbm_strong_errors

h, errors, order = gbm_strong_errors(
    "milstein",
    step_counts=[8, 16, 32, 64, 128],
    paths=20_000,
)
print(order)
```

### Rare-event importance sampling

```python
from stochastic_lab import normal_tail_probability_importance_sampling

estimate, se = normal_tail_probability_importance_sampling(
    threshold=4.0,
    simulations=200_000,
    seed=1,
)
```

### Correlated Brownian motion

```python
import numpy as np
from stochastic_lab import correlated_brownian_motion

corr = np.array([[1.0, 0.7], [0.7, 1.0]])
_, paths = correlated_brownian_motion(corr, paths=10_000, seed=42)
```

### PCA factor decomposition

```python
from stochastic_lab import pca_factor_decomposition, covariance_explained

fit = pca_factor_decomposition(data, n_components=3)
print(covariance_explained(fit["eigenvalues"], 3))
```

### Vasicek yield curve

```python
import numpy as np
from stochastic_lab import vasicek_yield_curve

maturities = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
yields = vasicek_yield_curve(
    short_rate=0.03,
    maturities=maturities,
    kappa=1.2,
    theta=0.045,
    sigma=0.012,
)
```

### Heston option pricing

```python
from stochastic_lab import heston_european_call_mc

price, se = heston_european_call_mc(
    s0=100.0,
    strike=100.0,
    rate=0.02,
    maturity=1.0,
    v0=0.04,
    kappa=2.0,
    theta=0.04,
    xi=0.30,
    rho=-0.7,
    paths=50_000,
    seed=7,
)
```

## Repository structure

```text
src/stochastic_lab/
    processes.py          core diffusion and counting processes
    schemes.py            Euler-Maruyama and Milstein
    convergence.py        strong/weak convergence diagnostics
    jumps.py              compound Poisson and Merton jump diffusion
    monte_carlo.py        estimators and variance reduction
    rare_events.py        importance sampling
    qmc.py                Sobol and randomised QMC
    estimation.py         GBM and OU estimation
    calibration.py        fit diagnostics and parameter checks
    heston.py             stochastic-volatility simulation
    heston_pricing.py     Heston MC pricing and calibration losses
    regime_switching.py   Markov regimes and state-dependent returns
    multivariate.py       correlated stochastic processes
    factor_models.py      PCA factor decomposition
    term_structure.py     Vasicek short rates, bond prices and yields
    pricing.py            option-pricing applications
    risk.py               VaR, ES and drawdown analytics
experiments/
    advanced_models_demo.py
    factor_term_structure_demo.py
tests/                    statistical and numerical verification
.github/workflows/        CI across Python 3.10-3.12
```

## Development roadmap

1. **Foundations** — Brownian motion, GBM, OU, CIR and Poisson. ✅
2. **Numerical SDE analysis** — Euler-Maruyama, Milstein, strong/weak convergence. ✅
3. **Jump processes** — compound Poisson and Merton jump diffusion. ✅
4. **Variance reduction** — antithetic and control variates. ✅
5. **Importance sampling** — rare-event estimators. ✅
6. **Quasi-Monte Carlo** — Sobol and randomised QMC. ✅
7. **Parameter estimation** — GBM MLE and OU estimation. ✅
8. **Stochastic volatility** — Heston simulation and diagnostics. ✅
9. **Regime switching** — Markov state processes and switching returns. ✅
10. **Applications** — pricing, risk, term structure, multivariate simulation and factor modelling. ✅ Core implemented
11. **Research experiments** — controlled comparisons of convergence, parameter recovery, factor compression and model behaviour. 🚧 Active
12. **Advanced extensions** — Brownian bridge/QMC path construction, CIR bond pricing, multi-factor rates, Heston Fourier pricing and richer calibration workflows. Planned

## Design principles

- mathematical transparency over black-box output;
- explicit random seeds and reproducibility;
- reusable package code rather than notebook-only implementations;
- analytical benchmarks wherever available;
- numerical convergence and estimation diagnostics;
- uncertainty reporting through standard errors;
- testable model assumptions and parameter constraints;
- documented approximations and limitations.

## Disclaimer

This repository is an independent educational and research project. It is not financial or investment advice, and its models should not be used for production decisions without appropriate validation, governance and controls.
