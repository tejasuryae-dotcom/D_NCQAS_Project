# Revision 3: development and held-out analysis protocol

This protocol was written during the revision, before launching any final
evaluation seed. It is a local prospective analysis plan, not an externally
registered preregistration. Earlier version-2 results informed the questions.

## Questions and endpoints

1. Does the selected D-NCQAS implementation improve held-out tail fidelity over
   a frozen policy, domain-randomized PPO, categorical cross-entropy search,
   and aging evolution at the specified proposal budget?
2. Does changing the optimizer produce measurable policy change, and does that
   change improve typical sampled circuits or only the best selected circuit?
3. Are the molecular results sensitive to optimization caps, and can a separately
   checked parameter-shift optimizer reach a declared stationarity criterion?

The primary endpoint is per-seed OU premeasurement GHZ fidelity CVaR(0.20),
averaged with equal weights over three fixed tasks: 3-qubit tree, 5-qubit tree,
and 5-qubit chain. The independent unit is a training seed, not a task or a noisy
draw. Premeasurement fidelity is not readout robustness. The two five-qubit
graphs differ; the redundant three-qubit chain is excluded.

## Development and selection

- Development seeds: 100–103; no seed from the final set is used for tuning.
- Optimizer candidates: SGD at 0.003; Adam at 0.001 and 0.003.
- Development tasks: 3-qubit and 5-qubit trees. Each setting uses 16 generations
  of 16 proposals, 20 training-noise draws, and the same 20-draw validation selector.
- Selection criterion: highest mean candidate-validation CVaR over development
  tasks and seeds. Final test fidelity is not a tuning criterion.
- CEM gets three elite fractions (0.125, 0.25, 0.5), smoothing 0.7 and positive
  pseudocount 0.01. Aging evolution gets tournament sizes 2, 4 and 8. Each gets
  the same number of development configurations and task/seed runs as PPO.
- Selected settings are written to a machine-readable selection record before
  the final manifest is created. No selection changes are allowed after that.
- Budget scaling is development-only: 144, 384 and 768 proposals, comparing
  the selected learned policy with its frozen-policy control.

## Final comparison

Use 32 fresh seeds (1000–1031), three tasks, 100 independent scenarios per
nominal/static/OU regime, and 384 proposals per search method (24 generations
of 16). All search methods share the legal discrete gate alphabet, minimum
length, adjacent inverse-gate mask, maximum length 2n+2, and validation selector.

Methods: D-NCQAS, No-Learning, Domain-Random, Nominal-PPO, CEM, Age-Evolution,
and Validation-Tree. The tree control enumerates roots and is not forced to
spend the search budget. All actual circuit-evaluation counts and timings are
reported. Equal proposals do not mean equal simulator calls or wall-clock costs.
Initial/final policy probes use 32 samples each and are diagnostics outside the
search budget; their 64 full-circuit-equivalent evaluations are separately stated.

CEM and aging evolution are independent adaptations of generic published
optimization principles. They are not vendor baselines or reproductions of
state-of-the-art quantum software. Their fixed training OU bank is independent
of validation/testing. They score a nominal/robust mixture with the same
complexity formula; D-NCQAS additionally learns its adversary. These comparisons
evaluate entire methods. The Domain-Random contrast isolates the adversary more
closely. Frozen-policy initialization and architect sampling streams match PPO.

## Inference and power planning

Four primary two-sided paired sign-flip tests compare D-NCQAS with No-Learning,
Domain-Random, CEM and Age-Evolution. Holm correction controls this fixed family
at 0.05. At 32 seeds the implementation uses 100,000 random sign vectors with
the plus-one Monte Carlo correction and a fixed analysis RNG seed. Per-task,
Nominal-PPO, tree-control and ablation contrasts are secondary and labeled as such.

Planning assumes a meaningful paired difference of 0.075 fidelity units and a
paired standard deviation of 0.12. A noncentral paired-t calculation at the
conservative Bonferroni threshold 0.05/4 gives approximately 80.5% power with
32 pairs. This is an explicit planning model, not a guarantee for the sign-flip
test or for smaller effects. Report the achieved standard deviation and intervals;
do not substitute post-hoc observed power. The old ten-pair resolution barrier
does not apply to this design.

Seed bootstrap confidence intervals target the same aggregated endpoint. Report
both pointwise 95% and Bonferroni-level 98.75% percentile intervals as approximate
bootstrap intervals. Do not equate a nonsignificant result with equivalence.
Sign-flip inference assumes symmetry/exchangeability of null differences.

Four single-component ablations (No-Spatial, No-Temporal, No-CVaR, No-Complexity)
use the first 16 final seeds on the five-qubit tree only. Their separate four-test
Holm family is secondary/exploratory, without a confirmatory power claim.

## Molecular convergence protocol

Retain the traceable H2, HeH+ and LiH compact CAS(2,2) matrices. Use ten fresh
seeds, one/two layers, clean/nominal training and five matched starts. Compare
COBYLA caps 400 and 2000, then BFGS with an independently tested parameter-shift
Jacobian and maximum 2000 iterations from the same starts. This final algorithm
is specified before looking at held-out noisy energies.

Select the lowest training energy among the five BFGS runs. Report both the
solver status and an independent stationarity check, maximum absolute gradient
at most 1e-6 Ha/radian. Stationarity is not a proof of global optimality. Freeze
the selected angles for clean evaluation and 100 draws per noisy regime.
Preserve all capped, unsuccessful and successful runs. Do not discard failed
starts or select hyperparameters based on test energy. VQE remains a separate
fixed-ansatz validation experiment, not learned VQE architecture search.

## Traceability and claims

Store configuration/source hashes, job identity, trained parameters, diagnostic
histories, scenarios, raw observations, optimizer traces and all analysis code.
Do not infer hardware validation, mitigation, universal advantage, or broad
novelty from these synthetic small-system experiments. Rewrite the paper around
the actual optimizer, selection and convergence findings, including null results.
