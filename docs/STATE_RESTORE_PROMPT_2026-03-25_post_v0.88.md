# State Restore Prompt (Post v0.88, 2026-03-25)

1. Load snapshot:
- `docs/SYSTEM_STATE_SNAPSHOT_2026-03-25_post_v0.88.json`

2. Current release state:
- Version is `0.88.0` in both package and pyproject.
- Artifact persistence and run-id bootstrap are wired for production notebooks.
- Baseline workflow is load-or-train by default in backend baseline runs.
- Calibration defaults to full scoring loader when no batch cap is set.

3. Validation baseline:
- `python -m pytest -q tests/test_notebook_experiments.py -q`
- `python -m pytest -q tests/test_v7_dual_framework.py -q`
- `$env:PYTHONPATH='src'; python -m pytest -q tests/test_viz_persistence.py -q`

4. If context resets:
- Reopen this file and the post snapshot JSON first, then continue from release hardening or notebook UX improvements.
