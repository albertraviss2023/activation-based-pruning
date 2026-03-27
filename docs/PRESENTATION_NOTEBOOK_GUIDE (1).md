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

## Notes

- Presentation Mode auto-resolves common variable names from notebook globals.
- If resolution fails, run the main workflow cells first.
- Some setup cells use notebook magics (`!pip`, `%autoreload`) and are expected to run in notebook environments.
