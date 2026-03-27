# ReduCNN Workflow How-To Guide (v0.88)

This guide documents the core end-to-end workflows for research, demos, and thesis reporting.

It assumes:
- You cloned the repo from GitHub.
- You run from VS Code locally or Google Colab.
- You want reproducible artifacts saved to disk (models, plots, HTML/GIF visuals, CSV tables).

## 1. Environment Setup (GitHub + VS Code + Colab)

### 1.1 Clone and install (local VS Code)

```bash
git clone https://github.com/albertraviss2023/activation-based-pruning.git
cd activation-based-pruning
pip install -e .
```

### 1.2 Clone and install (Google Colab)

```bash
!git clone https://github.com/albertraviss2023/activation-based-pruning.git
%cd activation-based-pruning
!pip install -e .
```

### 1.3 Artifact and model locations

Use these as the default mental model across notebooks:
- Baselines: `saved_models/baselines/<backend>/<dataset>/<model>/...`
- Raw pruned: `saved_models/pruned_raw/<backend>/<dataset>/<model>/<method>/...`
- Fine-tuned: `saved_models/fine_tuned/<backend>/<dataset>/<model>/<method>/...`
- Run artifacts (plots/HTML/GIF/CSV): `outputs/experiments/<dataset>/<model>/<run_id>/...`
- Artifact mirror: `saved_models/artifacts/<dataset>/<model>/<run_id>/...`

## 2. Workflow A: Register Custom Math and Use It on Pretrained Models

### 2.1 Notebook to run

- `experiments_custom_method_registration_minimal RESNET_18.ipynb`
- `experiments_custom_method_registration_minimal_vgg16.ipynb`
- `experiments_custom_method_registration_minimal densenet121.ipynb`
- `experiments_custom_method_registration_minimal mobilenet_v2.ipynb`

Choose the one matching your target model.

### 2.2 Canonical registration style

Use framework-specific decorators and backend tools:

```python
@register_method('custom_l2', framework='torch')
@register_method('custom_l2', framework='keras')
def custom_l2_score(layer, **kwargs):
    tools = kwargs['tools']
    mode = str(kwargs.get('l2_mode', 'sum')).lower().strip()
    return np.asarray(tools.weight_l2(layer, mode=mode), dtype=np.float64).reshape(-1)

@register_method('chip', framework='torch')
@register_method('chip', framework='keras')
def chip_score(layer, **kwargs):
    tools = kwargs['tools']
    A, _ = tools.collect_layer_outputs(layer, max_batches=kwargs.get('calib_batches'), include_labels=False)
    if A is None:
        return None
    return np.asarray(tools.chip_scores(A, max_spatial=kwargs.get('chip_max_spatial', None)), dtype=np.float64).reshape(-1)
```

### 2.3 What this workflow validates

- Registration API compatibility.
- Cross-backend execution (PyTorch and Keras).
- Scoring robustness across model families.
- Checkpoint creation for raw-pruned and healed models.

## 3. Workflow B: Load Pretrained, Prune, Heal, Visualize, Save

### 3.1 Notebook to run

- `experiments_pretrained_workflow.ipynb`

### 3.2 Minimal knobs to set

Set these config values near the top:
- `BACKEND` (`'pytorch'` or `'keras'`)
- `MODEL_TYPE` (`'resnet18'`, `'vgg16'`, `'densenet121'`, `'mobilenet_v2'`)
- `PRUNE_METHOD` (bundled or registered method)
- `PRUNE_SCOPE` (`'local'` or `'global'`)
- `PRUNE_RATIO`
- `FINETUNE_EPOCHS`

### 3.3 Behavior

The notebook will:
1. Load baseline checkpoint if available.
2. Train baseline only if missing.
3. Prune using selected method.
4. Save raw-pruned checkpoint.
5. Fine-tune healed model.
6. Save healed checkpoint.
7. Produce metrics, comparisons, and visualization outputs.

### 3.4 Outputs to expect

- Saved models in `saved_models/...`
- Visuals in `outputs/...`
- Comparison plots and inference gallery
- Structured tables for reporting

## 4. Workflow C: Train New Baseline, Then Prune + Heal + Report

### 4.1 Notebooks to run

Production notebooks:
- `experiments_cifar10.ipynb`
- `experiments_cifar100.ipynb`
- `experiments_cat_dog.ipynb`
- `experiments_cifar100_github.ipynb`
- `experiments_cat_dog_github.ipynb`

### 4.2 What is already standardized in these notebooks

- Unified custom method registration for `custom_l2`, `chip`, `custom_nisp`, and `custom_spectral_energy`.
- Framework-specific decorators for registration.
- Run-id artifact folders.
- End-of-run CSV export cell for tabular results.

### 4.3 CSV persistence (important for thesis reporting)

At the end of each production notebook, run the final export cell:
- It scans result-like variables (`result`, `summary`, `report`, `metric`, `table`, `record`, `comparison`).
- It writes CSV files to `outputs/.../tables`.
- It mirrors CSV files to `saved_models/artifacts/.../tables`.

This avoids re-running experiments just to rebuild tables.

## 5. Workflow D: Visualization Deep Dive (Narrative Demo)

### 5.1 Notebook to run

- `visualization_deep_dive.ipynb`

### 5.2 Supported modes

- Use loaded pretrained baselines when available.
- Train baselines if missing (based on notebook mode setting).
- Run pruning surgery and recovery.
- Generate deep visual diagnostics and presentation-ready artifacts.

### 5.3 Typical deep-dive artifacts

- Network graph HTML
- Candidate discovery graph HTML
- Pruning process HTML/GIF
- Architecture comparison HTML
- Method battle and surgery animations
- Activation flow diagrams

These are written under `outputs/<run_id>/...` with latest mirrors for quick presentation access.

## 6. Recommended Run Discipline

1. Set a unique `RUN_ID` per experiment run.
2. Keep one backend/model/method combination per run when reporting results.
3. Do not delete `saved_models/artifacts/...` between runs.
4. Use the final CSV export cell before ending the session.
5. Commit notebooks and generated tables/plots metadata needed for reproducibility.

## 7. Quick Troubleshooting

1. Method not found:
Verify the registration cell executed before pruning, and verify each decorator has the correct framework (`framework='torch'` or `framework='keras'`).

2. No CSV exported:
Ensure the final export cell ran, and ensure result variable names include reporting keywords (or use pandas DataFrames directly).

3. Baseline mismatch or reload issues:
Confirm backend/model/dataset folder alignment in `saved_models/baselines/...`.

4. Slow runs:
Use a faster model first (`resnet18`) for pipeline validation, then run heavier models.
