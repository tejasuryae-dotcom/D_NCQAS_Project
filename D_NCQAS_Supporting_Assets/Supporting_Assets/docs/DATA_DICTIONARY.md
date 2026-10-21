# Reading the revision-3 results

All paths below are relative to `outputs/`. A job is one method/task/training-seed
run; a noise observation is a repeated evaluation of its selected fixed circuit.
Noise observations are not independent training replicates.

| Location | Contents and analysis unit |
|---|---|
| `development_optimizer/` | 24 runs: three settings, two tasks, four development seeds |
| `development_baselines/` | 48 runs: two algorithms, three settings each, two tasks, four seeds |
| `development_scaling/` | 36 runs: two methods, three budgets, two tasks, three seeds |
| `final_study/` | 672 main runs and 64 secondary ablation runs |
| `vqe_convergence/` | 120 molecular fits, each with three optimizer branches and five starts |
| `analysis/tables/primary_seed_matrix.csv` | 32 rows; each method column averages its CVaR equally over three fixed tasks |
| `analysis/tables/primary_tests.csv` | Four prespecified paired primary comparisons |
| `analysis/tables/secondary_task_tests.csv` | Eighteen task-specific contrasts with one separate Holm family |
| `analysis/tables/ablation_tests.csv` | Four secondary contrasts, each with 16 matched seeds |
| `analysis/tables/ghz_seed_results.csv` | One row per study/job/regime, including development; filter `phase=final` for main inference |
| `analysis/tables/policy_diagnostics.csv` | Initial/final probe means, parameter change, KL and exact rollout-probability checks |
| `analysis/tables/learning_history.csv` | Generation-level records; fields depend on the method |
| `analysis/tables/vqe_optimizer_runs.csv` | 1,800 start records: 120 fits × three branches × five starts |
| `analysis/tables/vqe_seed_results.csv` | 480 fit/regime summaries from final BFGS-selected parameters |
| `analysis/raw/ghz_observations.csv` | 220,800 main/ablation observations; development raw observations remain in job JSONs |
| `analysis/raw/vqe_observations.csv` | 36,120 observations: one clean and 100 per noisy regime for each fit |
| `analysis/raw/scenarios.jsonl` | 28,800 shared GHZ noise configurations, once per task/seed/regime/draw |

## Identifiers and units

- `id` identifies a GHZ job. `seed` controls independent training/random streams.
  Main seeds are 1000–1031; ablations use 1000–1015; development uses 100–103.
- `topology` and `n_qubits` jointly identify the GHZ task. A three-qubit tree is
  chain-shaped; the two five-qubit graphs differ.
- `scenario` is a within-regime integer. `scenario_id` hashes the complete noise
  configuration. Pairing uses task/seed/regime/scenario, not circuit length.
- `fidelity` is premeasurement GHZ state fidelity in [0,1]. `p_ghz` is the
  assignment-noisy probability of observing all zeros or all ones; it is not
  a coherent-state fidelity or an entanglement witness.
- `fidelity_cvar20` is the mean of the worst 20 percent empirical fidelity mass.
  Larger is better. Test CVaR and candidate `validation_cvar` use disjoint banks.
- `delta` is full D-NCQAS minus the named comparator. A positive value favors
  D-NCQAS. `p_signflip` is two-sided. `p_holm` covers the four primary tests;
  `p_holm_secondary` covers only the explicitly named secondary family.
- `ci95_*` are pointwise seed-bootstrap bounds. `ci9875_*` use a 98.75 percent
  level for the four primary contrasts; they are approximate percentile intervals.
- `sim_calls_train` counts search and candidate-validation circuit evaluations.
  Initial/final policy probes add 64 separately declared full-circuit-equivalent
  evaluations. Analytic trees do not spend the 384-proposal search budget.
- `train_seconds` is elapsed time during parallel execution; it is not a
  controlled single-process benchmark. Cache reuse and unique-candidate counts
  differ between algorithms.
- VQE energies and errors are in hartree. `energy_error` subtracts the saved
  active-space FCI reference from the physical expectation. Assignment-noisy
  `measured_energy` is separate and has no guaranteed variational bound.
- `gradient_inf` is the maximum absolute parameter-shift derivative in
  Ha/radian. `stationary` means at most 1e-6; `success` is the solver's status.
  Neither proves a global minimum. `selected` marks the lowest training energy
  among five starts, independently within each optimizer branch.

## Provenance and interpretation

Per-job JSONs preserve full circuits, observations, relevant trained parameters,
random states, histories and optimization statuses. Phase manifests record the
planned jobs. The final protocol hash remains unchanged. Any release-only version
metadata migration is explicitly recorded; numerical source changes are rejected.

The integrity checker validates all job identities and summaries, paired scenario
IDs, all 120 VQE finite-difference stationarity checks, and selected numerical
replays. It does not prove every scientific claim or validate a hardware model.
See `REVISION_PROTOCOL.md` and the manuscript for the hypotheses and scope.
