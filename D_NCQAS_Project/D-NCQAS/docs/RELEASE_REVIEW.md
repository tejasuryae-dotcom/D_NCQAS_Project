# Revision-3 technical review

**Submission verdict: Needs Minor Revisions within the stated diagnostics/reproducibility scope.**
Computational corrections and verification are complete; authorship, repository
deposition and venue-specific formatting/declarations remain to be finalized.
Acceptance and novelty requirements remain the venue's decision.

The requested follow-up work is implemented: learning diagnostics and optimizer
selection, development-only budget scaling, tuned independent generic search
controls, a frozen fresh-seed design, matched VQE cap/refinement experiments,
and a data-generated manuscript.

The completed artifact is suitable for technical review and a scoped reproducibility
or diagnostics submission after author and venue finalization. It must not be
presented as proof of state-of-the-art, hardware or general quantum advantage.
Venue-specific novelty/acceptance cannot be established by the test suite.

## Evidence and scope

- 20 numerical/regression tests pass in a fresh locked environment.
- Five notebook code cells pass sequentially in-process; live Colab was not tested.
- 108 development jobs remain separate from 672 main and 64 secondary-ablation jobs.
- The main inference uses 32 seed pairs, three fixed tasks and four Holm-adjusted
  comparisons; it removes the former ten-pair testing-resolution barrier.
- The power model targets a 0.075 effect with assumed paired SD 0.12. It is not
  guaranteed power for smaller effects, the exact achieved distribution, or arbitrary tasks.
- CEM and aging evolution are independently implemented generic controls. They
  do not constitute reproduction of every specialized published QAS baseline.
- All 120 selected BFGS VQE fits satisfy the independently checked gradient threshold;
  112 report solver success. Stationarity is distinct from global optimality.
- Complete results and adjusted effects are in outputs/analysis/tables/primary_tests.csv;
  the manuscript interprets them without suppressing null or negative outcomes.

## Main findings

The only significant primary contrast is against No-Learning: mean difference
+0.0322, Holm-adjusted p=0.0471. There is no significant primary
advantage over Domain-Random, CEM or Age-Evolution. Mean task-averaged tail
fidelity is 0.5049 for D-NCQAS and 0.7865 for the analytic tree. The revised
paper presents these limits directly. Correct gradients and measurable policy
change do not establish effective GHZ synthesis or an adversarial advantage.

## Remaining boundaries

Graphs/rates remain synthetic; no hardware, readout-mitigation or finite-shot
performance is established. The 3/5-qubit study is limited in scale. VQE remains
a compact fixed-ansatz CAS(2,2) validation, not learned molecular architecture
search or a full-molecule LiH calculation. Results are conditional on the selected
configuration, finite budget, three tasks and scenario distribution.

The source is modular research code with raw-data provenance, deterministic job
recovery, exact dependency pins and verification scripts. It is not a production
API with comprehensive cross-platform certification. The final verification report,
not a prose status label, is the authority for completed experiment counts.

See manuscript/REVISION_RESPONSE.md for the requested-work mapping,
REVISION_PROTOCOL.md for the frozen design, and SUBMISSION_CHECKLIST.md for the
remaining author/venue actions. No version-2 results are relabeled as new data.
