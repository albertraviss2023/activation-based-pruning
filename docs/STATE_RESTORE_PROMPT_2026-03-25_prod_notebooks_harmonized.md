# State Restore Prompt (Production Notebooks Harmonized, 2026-03-25)

1. Load snapshot:
- `docs/SYSTEM_STATE_SNAPSHOT_2026-03-25_prod_notebooks_harmonized.json`

2. What is done:
- Unified custom registration in all production notebooks using framework-specific decorators.
- Registered methods: `custom_l2`, `chip`, `custom_nisp`, `custom_spectral_energy`.
- Added CSV artifact helper + end-of-notebook auto-export cell.
- CSV files persist under run-id artifact folders and mirror to `saved_models/artifacts/...`.
- Disabled legacy conflicting registration blocks in `experiments_cifar10.ipynb`.

3. Validation command:
- `python -m pytest -q tests/test_notebook_experiments.py -q`

4. If context resets:
- Open this prompt and the snapshot JSON first, then continue from notebook execution checks / additional UX cleanup.
