# Stage 9: weak Heston PINN confirmation protocol

Status: prospective protocol; no new training results claimed.

## Scientific question
Does weak-form enforcement improve independent Heston option-price accuracy relative to a matched strong-form PINN, after adequate quadrature resolution and equal computational budgets?

## Prior pilot (unconfirmed)
Historical Experiment 014 reported price MAE 1.7455 versus 1.3040 (~25.3% improvement) and weak residual ~10.1 versus ~2.5. A subsequent audit found only ~1.3% improvement on an original evaluation grid and major quadrature sensitivity (~1.19e-6 at order 12 versus 0.16174 at order 48). Treat as exploratory only; verify against original manifests.

## Prerequisite audit
1. Locate and preserve the original PINN implementation, frozen configurations, checkpoints, training manifests, reference prices, and per-seed results. Do not infer that files absent from this repository never existed.
2. Verify PDE coefficients, time convention, variance boundary treatment, payoff and all imposed hard constraints.
3. Verify weak-form integration-by-parts boundary terms and test-function regularity.
4. Record hardware, software versions, numerical precision, seeds, wall time, sample counts, and quadrature nodes/weights.
5. Verify that benchmark prices converge independently of neural training.

## Confirmatory design
- Compare strong-form and weak-form objectives under the same architecture, parameter regimes, payoff conditions, random seeds, collocation budget and measured wall-clock/compute budgets.
- Preselect at least five paired seeds; report all seeds, not just best runs.
- Train weak objectives at quadrature orders 12, 24, 36 and 48 (or justify a different convergent sequence). Evaluate all checkpoints using a higher-order independent quadrature rule, e.g. 64 and 80, and check stability. Quadrature order alone does not guarantee convergence.
- Keep evaluation points and benchmark reference prices independent of training points and training quadrature nodes.
- Report price MAE, RMSE, max error, Delta/Gamma error, strong residual, independent weak residual, constraint violations, wall-clock time and peak memory.
- Report paired per-seed differences, means, dispersion, confidence intervals and complete failure cases.
- Use common validation rules and fixed stopping criteria. Record hyperparameter search budgets and do not tune on the held-out test set.

## Gates
A. Mathematical verification: correct PDE and weak form including boundary contributions.
B. Numerical verification: independent weak residual stable under quadrature refinement.
C. Empirical verification: improvement on independent price benchmarks and multiple paired seeds.
D. Economic verification: Greeks, no-arbitrage properties and compute cost not materially degraded without explicit trade-off.
E. Generalisation: replicate across preselected Heston parameter regimes.

## Decision rules
- If A or B fails, fix formulation or quadrature before making performance claims.
- If A/B pass but C fails, report a well-supported negative result.
- If A-C pass but D fails, describe the trade-off, not unqualified superiority.
- Only after A-E pass consider the weak-form improvement confirmed and proceed to higher-dimensional extensions.

## Next implementation task
Recover the exact Stage 9 code and manifests; add unit tests for manufactured smooth solutions, quadrature refinement and boundary terms before launching the five-seed training matrix.

## Reproducibility rule
Never overwrite pilot artifacts. Preserve configurations, hashes, raw outputs and scripts needed to regenerate every thesis table and figure.
