# Contributing

Use Python 3.12 and the lockfiles. Run `python -m unittest discover -s tests -v`.
For scientific changes, add a numerical/reference regression test, execute the smoke
pipeline, and invalidate/recompute affected results. Never hand-edit result tables.
Keep method controls and scenario pairing fixed; report negative results. Use a new
run directory and update the protocol/version when changing hypotheses or metrics.
