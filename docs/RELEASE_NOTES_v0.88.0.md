# Release Notes - v0.88.0

Date: 2026-03-25

## Summary
v0.88.0 focuses on production workflow quality for research notebooks:
- scoring defaults now reflect real pruning workflows (full calibration pass unless explicitly limited),
- baseline handling is load-or-train by default,
- experiment plots and visual outputs are persistable by default via run-scoped artifact directories.

## Why This Release
The goal is to keep demo notebooks minimal while preserving research rigor:
- fewer confusing knobs in production flows,
- deterministic artifact capture for thesis writing and presentations,
- backend-consistent behavior across PyTorch and Keras.

## Major Changes

### 1) Production calibration default
Scoring no longer assumes a small fixed calibration batch count unless explicitly requested.

Behavior:
- Explicit user setting present (`prune_batches`, `calib_batches`, `calibration_batches`) -> honored.
- No setting present -> full calibration loader length.
- Non-sized/infinite loader -> safe fallback cap (`prune_batches_fallback=128`).

Updated components:
- `src/reducnn/backends/torch_backend.py`
- `src/reducnn/backends/keras_backend.py`
- `src/reducnn/pruner/custom_method_tools.py`

### 2) Baseline policy: load-or-train
Baseline training calls now support non-negotiable load-or-train behavior.

Behavior:
- If baseline checkpoint exists -> auto-load and skip retraining.
- If missing -> train and auto-save baseline checkpoint.
- Works in both backends.

Checkpoint roots:
- PyTorch: `saved_models/baselines/pytorch/<dataset>/<model>/...`
- Keras: `saved_models/baselines/keras/<dataset>/<model>/...`

### 3) Artifact persistence for thesis/presentation workflows
Visualization persistence now has a single activation mechanism using environment variables.

Environment variables:
- `REDUCNN_ARTIFACT_DIR` (primary save path)
- `REDUCNN_ARTIFACT_MIRROR_DIR` (optional mirror path)
- `REDUCNN_RUN_ID` (filename prefix)
- `REDUCNN_DATASET_KEY` (used by baseline pathing fallbacks)

What auto-persists:
- Matplotlib plots from stakeholder and research visualization modules.
- Plotly renders in animation workflows.

### 4) Production notebook updates
Updated notebooks:
- `experiments_cifar10.ipynb`
- `experiments_cifar100.ipynb`
- `experiments_cat_dog.ipynb`
- `experiments_cifar100_github.ipynb`
- `experiments_cat_dog_github.ipynb`

Injected a run bootstrap cell to:
- create run-scoped outputs directory,
- set artifact environment variables,
- mirror artifacts into `saved_models/artifacts/...`.

## Compatibility Notes
- Existing code that explicitly sets calibration batch limits keeps current behavior.
- Existing notebooks continue to run; artifact saving now activates automatically once the run bootstrap cell sets env vars.
- If you want to disable baseline auto-load behavior for a specific workflow, set:
  - `baseline_checkpoint_policy='off'` in adapter config.

## Versioning
- Package version bumped to `0.88.0` in:
  - `pyproject.toml`
  - `src/reducnn/__init__.py`
