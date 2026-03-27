# Visualization Review Summary (2026-03-22)

## Project Context
This review covered the full `activation-based-pruning` repository with emphasis on pruning-process visualization quality for presentations, especially:
- candidate discovery after metric scoring (e.g., APoZ),
- animated pruning execution (what is pruned vs kept),
- final pruned architecture graph by model type (VGG, ResNet, DenseNet).

## What Was Reviewed
- Core pruning pipeline:
  - `src/reducnn/pruner/surgeon.py`
  - `src/reducnn/pruner/mask_builder.py`
- Dependency tracing and structural surgery:
  - `src/reducnn/backends/torch_backend.py`
  - `src/reducnn/backends/keras_backend.py`
- Visual systems:
  - `src/reducnn/visualization/animator.py`
  - `src/reducnn/visualization/flow_animator.py`
  - `src/reducnn/visualization/pruning_visualizer.py`
- Notebook presentation integration and artifacts:
  - `docs/PRESENTATION_NOTEBOOK_GUIDE.md`
  - presentation sections in experiment notebooks
  - existing outputs in `outputs/` and `exports/`

## Confirmed Current Pipeline
`trace_graph -> score_map -> build_pruning_masks -> apply_surgery`

- Residual dependency clusters are traced and harmonized.
- Local and global pruning both exist; global is cluster-aware.
- Multiple visualization systems exist but are partially overlapping and not fully narrative-aligned.

## Key Findings
1. Strong backend dependency logic already exists.
- ResNet cluster constraints are handled in both tracing and surgery.
- Concatenative models (DenseNet) are handled with concat channel-offset logic.

2. Visualization quality gap is mostly storytelling, not missing raw data.
- Current visuals are often layer-level summaries; channel-level candidate reasoning is weak.
- Discovery stage is mostly static coloring, not an explicit dependency sweep narrative.
- Surgery animation does not clearly show candidate -> cut -> retained in a synchronized graph-level view.
- Final before/after architecture comparison is not yet a polished side-by-side structural story.

3. Notebook "Presentation Mode" exists and is a good insertion point.
- This can be upgraded without redesigning the full training/pruning flow.

## Model-Type Notes
- VGG-style sequential models: best shown as linear cascade with channel-strip shrink.
- ResNet residual models: must visually expose cluster lock/harmonization constraints.
- DenseNet concatenative models: need block-collapsed graph abstraction to avoid edge clutter.

## Overhaul Plan (Proposed)
1. Create a unified trace artifact for visuals (`PruningTrace`).
- Include nodes/edges/clusters, per-channel scores, masks, harmonized masks, and before/after channel stats.

2. Build candidate-discovery view.
- Color states for kept/candidate/protected/cluster-tied channels.
- APoZ/L1/Taylor switch.

3. Build animated surgery timeline.
- Score heat -> candidate pulse -> cut -> dependency propagation -> final settle.

4. Build final architecture comparison view.
- Side-by-side before vs after, synchronized layout, channel-aware node sizing, edge-width change.

5. Integrate into notebook Presentation Mode.
- Keep current flow; swap internals to trace-driven visual APIs.

6. Add visualization correctness tests.
- Cluster consistency, channel accounting, export validity.

## Immediate Implementation Priority
1. Implement `PruningTrace` data contract and export.
2. Implement first high-impact candidate graph for APoZ on ResNet/VGG.
3. Implement first surgery animation timeline using the same trace payload.

## Expected Outcome
Presentation-ready visuals that clearly communicate:
- why channels were selected as candidates,
- what was physically removed and what was retained,
- how dependencies constrained pruning decisions,
- and what the architecture changed to after surgery.
