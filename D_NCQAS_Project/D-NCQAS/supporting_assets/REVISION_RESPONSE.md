# Revision response

| Requested work | Completed revision | Evidence |
|---|---|---|
| Diagnose weak learning | Exact probability checks, KL/entropy/clipping/parameter diagnostics, initial/final probes, optimizer comparison, budget scaling | policy_diagnostics.csv; learning_history.csv; figures 1, 2, 5 |
| Resolve VQE caps | Matched COBYLA 400/2000 and BFGS parameter-shift study; all 120 selected fits stationary | vqe_optimizer_runs.csv; figure 6; independent Jacobian test |
| Strengthen controls/inference | Tuned generic CEM/evolution, frozen/domain-random/nominal controls, 32 fresh seeds, four planned primary tests, 16-seed secondary ablations | hashed REVISION_PROTOCOL.md; primary_tests.csv; ablation_tests.csv |
| Revise paper | New title, related work, full methods, power assumptions, fresh results, limitations and transparent scope | main_paper.pdf, main_paper.tex, main_paper.md |

Among four prespecified primary contrasts, 1 favor D-NCQAS and 0 favor a comparator after Holm correction. This is a controlled diagnostic study, not a demonstration of state-of-the-art or hardware superiority. Author/venue declarations remain for the submitting authors.
