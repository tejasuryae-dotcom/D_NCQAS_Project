# Quick start — revision 3

Extract `D_NCQAS_Revision3_Project.zip`, enter `D-NCQAS/`, and use Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-runtime.lock.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
```

## Fast execution check

```bash
python scripts/run_all_experiments.py --config configs/smoke.json --output outputs/my_smoke
python scripts/generate_figures_and_tables.py --output outputs/my_smoke
python scripts/verify_pipeline_integrity.py --output outputs/my_smoke
```

The smoke run checks execution; it does not generate the paper's research
results. All research outputs are already included in the archive.

## Reproduce every research phase

Use a new root (`outputs_reproduction/` below). Each phase recovers completed
per-job JSONs only when source/configuration matches. Worker count changes
execution concurrency, not study size or random seeds.

```bash
python scripts/revision_experiments.py --spec configs/development_optimizer.json --output outputs_reproduction/development_optimizer --workers 4
python scripts/revision_experiments.py --spec configs/development_baselines.json --output outputs_reproduction/development_baselines --workers 4
python scripts/revision_experiments.py --spec configs/development_scaling.json --output outputs_reproduction/development_scaling --workers 4
python scripts/revision_experiments.py --spec configs/final_study.json --output outputs_reproduction/final_study --workers 4
python scripts/vqe_convergence.py --output outputs_reproduction/vqe_convergence --workers 4
python scripts/analyze_revision.py --outputs outputs_reproduction
python scripts/verify_revision.py --outputs outputs_reproduction
python scripts/build_revision_paper.py --outputs outputs_reproduction
```

Final settings are frozen in `selected_settings.json` and `final_study.json`.
These commands reproduce those choices; they do not select a new configuration
from test data. Use a separately planned study if you change hypotheses, tasks
or tuning rules. The full suite may take hours on a small CPU. Avoid concurrent
writers to the same phase directory.

Manuscript compilation requires `pdflatex` with geometry, Latin Modern, graphicx,
booktabs, longtable, amsmath, amssymb, xcolor, hyperref, float and placeins. The PDF is already
supplied; LaTeX is not needed to run experiments.

## Optional tools

- Notebook: install `requirements-notebook.lock.txt` in addition to runtime
  dependencies, then open `notebooks/D_NCQAS_Execution.ipynb` in your frontend.
  Colab upload expects the complete revision-3 ZIP. The notebook runs tests and
  a fresh smoke pipeline and can display supplied research tables. Actual cells
  were validated in-process; live Colab was not tested.
- Chemistry: install `requirements-chemistry.lock.txt` and run
  `python scripts/build_hamiltonians.py`. Generated integrals/matrices are
  supplied and independently checked by tests.
- Packages are not vendored; initial installation requires network access.

The outputs ZIP holds all research phases and analyses. The supporting ZIP holds
setup, citation, licensing, protocol and review documents. The source ZIP is
required for execution. Read `docs/SUBMISSION_CHECKLIST.md` before submission.
