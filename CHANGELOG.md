# Changelog

All notable changes to this project are documented in this file.

## [0.88.0] - 2026-03-25

### Added
- Automatic artifact persistence support for visualization outputs:
  - New module: `reducnn.visualization.persistence`
  - Matplotlib plots now auto-save when `REDUCNN_ARTIFACT_DIR` is set.
  - Plotly renders auto-export HTML when `REDUCNN_ARTIFACT_DIR` is set.
  - Optional mirror path support via `REDUCNN_ARTIFACT_MIRROR_DIR`.
- Production notebook run bootstrap for artifacts:
  - `RUN_ID` initialization.
  - Output path creation under `outputs/experiments/...`.
  - Optional artifact mirror under `saved_models/artifacts/...`.

### Changed
- Calibration behavior for scoring is now production-first by default:
  - If no calibration batch limit is specified, full calibration loader length is used.
  - Explicit overrides still supported through:
    - `prune_batches`
    - `calib_batches`
    - `calibration_batches`
  - Safe fallback remains for non-sized loaders (`prune_batches_fallback`, default `128`).
- Baseline lifecycle policy in backends now supports mandatory load-or-train semantics for baseline runs:
  - Baseline train calls auto-load the latest saved baseline checkpoint if available.
  - If no baseline exists, baseline training proceeds and checkpoint is auto-saved.
  - Applies to both PyTorch and Keras adapters when training run name indicates baseline.
- Production notebooks (`cifar*`, `cat_dog*`) updated to include run-scoped artifact persistence bootstrap and to reduce default unsupported method friction.

### Fixed
- Realistic experiment workflows now persist visual outputs consistently for thesis and presentation workflows when artifact env vars are enabled.
- Cross-backend custom-method helper calibration behavior is now consistent with production defaults.

### Developer Notes
- Version bump:
  - `src/reducnn/__init__.py` -> `0.88.0`
  - `pyproject.toml` -> `0.88.0`

## [0.6.6] - Previous
- Prior baseline before v0.88 release hardening.
