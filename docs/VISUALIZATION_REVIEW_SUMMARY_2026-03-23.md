# Visualization Review Summary (2026-03-23)

## Scope
This snapshot captures the current state of the visualization overhaul for `visualization_deep_dive.ipynb`, focused on presentation-grade pruning storytelling with backend-agnostic execution (PyTorch and Keras).

## Milestone Status
- Overhaul implementation is active and mostly integrated.
- Notebook flow now supports backend injection and model fallback.
- Checkpoint handling has been simplified to a deterministic registry layout.
- Core visual outputs are generated with per-phase manifests.

## Implemented Fixes (Confirmed)
1. Backend-agnostic notebook imports and loading flow.
- The notebook no longer hard-requires torch in the core import cell.
- Data loading branches by backend: torchvision path for PyTorch, tf.data path for Keras.

2. Leakage guard in data splits.
- Pruning/visual calibration uses `val_loader`/validation split.
- Test split is preserved for final evaluation/inference only.

3. Deterministic model registry under `saved_models`.
- `saved_models/baselines/...`
- `saved_models/pruned_raw/...`
- `saved_models/fine_tuned/...`
- Timestamp-aware lookup via `CHECKPOINT_STAMP` and latest fallback.

4. Legacy checkpoint compatibility.
- Notebook policy cell includes legacy filename fallback discovery for prior single-file naming patterns.

5. Finetuned checkpoint loading fix for pruned architecture.
- Finetuned weights are loaded into `pruned_model` architecture, not into a fresh baseline-shaped model.
- This addresses VGG shape mismatch errors seen when loading pruned checkpoints into baseline models.

6. DenseNet crash guard in heavy visuals.
- Phase 3 skips heavy global GIF rendering when graph complexity is too high (node/edge thresholds), reducing notebook kernel OOM risk.

7. Output manifests per demo phase.
- Phase manifests are written (phase3, phase4, phase5) so users can reliably locate produced artifacts.

8. Readability upgrades in visual primitives.
- Activation flow supports threshold line and stronger below-threshold red highlighting.
- Network compression animation sorts layers alphabetically and shortens long labels.

## Verified Evidence
- Notebook compilation check across code cells: pass.
- Tests passing:
  - `tests/test_visualization_deep_dive_notebook_backend.py`
  - `tests/test_deep_dive_backend_agnostic.py`
  - `tests/test_viz_overhaul_story.py`
  - `tests/test_viz_deep_dive_all_models_matrix_schema.py`
- All-model matrix smoke report available at:
  - `outputs/viz_deep_dive/agnostic_validation/all_models_matrix_latest.json`
  - PyTorch pass: 4/4 (`resnet18`, `vgg16`, `densenet121`, `mobilenet_v2`)
  - Keras pass: 4/4 (same requested set via aliasing where needed)

## Current Demo Workflow (Simple Mode)
1. Select backend and model in policy/config cells.
2. Use `DEMO_MODE='pretrained_only'` when pretrained checkpoints exist.
3. Run phases in order (candidate discovery -> process animation -> surgery -> architecture comparison -> recovery/inference).
4. Use generated phase manifests to locate exact outputs and checkpoints used.

## Key Paths
- Notebook:
  - `visualization_deep_dive.ipynb`
- Core visualization modules:
  - `src/reducnn/visualization/pruning_visualizer.py`
  - `src/reducnn/visualization/research.py`
- Backend compatibility:
  - `src/reducnn/backends/keras_backend.py`
- Output roots:
  - `outputs/viz_deep_dive/`
  - `saved_models/`

## Open Items (Next Priority)
1. Validate all requested visuals are emitted for each model run and listed in manifests (especially architecture comparison, candidate discovery, pruning-process HTML/GIF variants).
2. Improve method battle and network surgery GIF legibility for dense graphs (label density and state persistence in final frames).
3. Re-check MobileNet recovery behavior for severe drops and tighten safe-mode heuristics only if needed.
4. Confirm activation-flow threshold styling is consistently visible across all backend/model runs.

## Continuity Notes
- The previous 2026-03-23 summary file was deleted and has now been recreated.
- Source-of-truth state for continuity is the JSON snapshot file for this same date.
