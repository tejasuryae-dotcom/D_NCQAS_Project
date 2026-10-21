# Revised study outputs

Each phase has a manifest, exact job inputs, completed per-job JSONs, logs and
RUN_COMPLETE. A job JSON contains the configuration, gates/parameters, diagnostics,
regime summaries and every raw observation. Completed jobs are immutable caches
for deterministic recovery; incomplete temporary files do not count as results.

- development_optimizer: 24 optimizer comparisons on development seeds.
- development_baselines: 48 CEM/evolution tuning runs.
- development_scaling: 36 learning/frozen-policy budget comparisons.
- final_study: 672 main runs plus 64 explicitly secondary ablations.
- vqe_convergence: 120 fixed-ansatz fits, each with 3 optimizer branches and 5 starts.
- analysis/tables: consolidated seed summaries, diagnostics and statistical tests.
- analysis/raw: GHZ/VQE observation CSVs and full shared GHZ scenario JSONL.
- analysis/figures: 600-DPI PNG and vector PDF plots generated from these tables.
- analysis/metadata: source, test, integrity, manuscript and execution evidence.

The independent primary inference unit is the seed, equally averaged over three
fixed tasks. The 220,800 GHZ observations include the secondary ablations; they
are not 220,800 independent training runs. VQE has 36,120 frozen-parameter observations.

Energy units are Hartree, timing units seconds, and gradient units Ha/radian.
Physical energy and assignment-noisy measured energy are different quantities.
Policy probes are unselected nominal circuit samples; selected-circuit validation
and held-out test scores are separately identified. A zero/negative finding is
retained. See the project protocol and manuscript for scientific interpretation.

See `../docs/DATA_DICTIONARY.md` for analysis units, field definitions, pairing,
cost accounting and the distinction between physical and measured energies.
Driver cache-guard receipts added during release verification are labeled as
post-run records; they must not be mistaken for prospective registration.
