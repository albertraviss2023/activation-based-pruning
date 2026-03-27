# Presentation Notebook Guide

This project now includes a standardized **Presentation Mode** section at the end of each experiment notebook:

- `experiments_cifar10.ipynb`
- `experiments_cifar100.ipynb`
- `experiments_cat_dog.ipynb`
- `experiments_cifar100_github.ipynb`
- `experiments_cat_dog_github.ipynb`
- `github experiments workflow_cifar10.ipynb`

## What Presentation Mode Shows

1. Dependency graph summary and cluster discovery
2. Pruning candidate discovery table (lowest-score channels)
3. X-ray graph walk:
   - Discovery
   - Importance
   - Consistency
   - Shrinkage
4. Feature-map visualization before/after pruning
5. Optional method agreement diagnostics

## New Presentation APIs (Overhaul)
Use `reducnn.visualization.animator.PruningAnimator`:

- `build_pruning_trace(model, score_map, masks, method_name, candidate_ratio)`
  - Produces a reusable trace artifact (nodes, edges, clusters, channel counts).
- `generate_candidate_discovery_graph(...)`
  - Interactive dependency graph + candidate table after metric scoring.
  - Candidate indices are interpreted from effective (cluster-harmonized) masks when masks are provided.
- `generate_pruning_process_animation(...)`
  - Stage animation: Discovery -> Candidates -> Cut -> Surgery -> Final.
- `generate_architecture_comparison(model, masks, method_name)`
  - Side-by-side architecture before vs after pruning.
- `export_pruning_trace(trace, path)`
  - Saves the trace artifact as JSON for handoff and reproducibility.

## GIF Readability Upgrades
- `GlobalFlowVisualizer(..., final_hold_frames=35)`
  - Keeps the final pruned state visible before loop reset.
- `GlobalMethodComparator(..., delta_mode=True)`
  - Highlights per-channel disagreements between two methods.
- `PruningVisualizer.animate_activation_flow(..., prune_ratio=..., threshold_mode='fixed')`
  - Adds a pruning-threshold line and red coloring below threshold.
- `PruningVisualizer.animate_pruning(..., order_mode='decision_then_score')`
  - Separates pruned vs kept zones to reduce confusion from cluster harmonization overrides.

## Recommended Demo Flow

1. Run notebook setup and dataset cells.
2. Run baseline + pruning cells until you have:
   - adapter
   - original model
   - pruned model
   - masks
   - data loader
3. Run **Presentation Mode** section at the end.
4. Optional:
   - set `EXPORT_XRAY_HTML = True` to save an embeddable HTML graph
   - set `RUN_HEAVY_DIAGNOSTICS = True` for method-comparison visuals
   - set `SAVE_PRUNED_CHECKPOINT = True` to export `pruned_model.(pth|weights.h5)`
   - set `COPY_TO_REPO_MODELS_DIR = True` to mirror checkpoints into `my_models/`

## Quick Overhaul Demo Export
From repo root:

```bash
PYTHONPATH=src python examples/overhaul_viz_demo.py
```

Exports:
- `outputs/overhaul_demo/pruning_trace.json`
- `outputs/overhaul_demo/candidate_discovery.html`
- `outputs/overhaul_demo/pruning_process.html`
- `outputs/overhaul_demo/architecture_comparison.html`

Interpretation tip:
- In local pruning with fixed ratio, candidate percentages can be uniform across layers.
- Use score distributions/rank correlation/decision agreement and architecture delta views for deeper insights.

## Notes

- Presentation Mode auto-resolves common variable names from notebook globals.
- If resolution fails, run the main workflow cells first.
- Some setup cells use notebook magics (`!pip`, `%autoreload`) and are expected to run in notebook environments.
