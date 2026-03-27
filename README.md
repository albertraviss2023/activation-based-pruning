# ReduCNN ✂️ (v0.88.0)

A professional, modular, dual-framework Python package for **Activation-Based Structural Pruning**. This package allows you to physically remove filters and channels from PyTorch and Keras models, reducing compute cost (FLOPs) and memory footprint while maintaining behavioral consistency.

## Installation

To install the package in editable mode (perfect for research and development):

```bash
git clone https://github.com/albertraviss2023/activation-based-pruning.git
cd activation-based-pruning
pip install -e .
```

---

## Workflow Documentation

For professional end-to-end usage guides (GitHub + VS Code + Colab), see:

- [Workflows How-To (v0.88)](docs/WORKFLOWS_HOWTO.md)

This includes:
- Registering custom method math and running on pretrained models.
- Loading pretrained baselines, pruning, healing, visualizing, and saving outputs.
- Training new baselines, then pruning/healing with full reporting.
- Visualization deep-dive workflow with artifact persistence.

---

## Sequential Decoupled Workflow

Unlike a monolithic script, `reducnn` is designed as a suite of independent tools. Here is how to use them sequentially:

### Step 1: Environment & Storage Setup (`sp.CloudStorage`)
**Why:** Researchers often switch between local VS Code and Google Colab. This module ensures your model paths remain consistent without changing code.

```python
import reducnn as sp

# project_name maps to a folder in your Google Drive 'MyDrive'
storage = sp.CloudStorage(project_name="MyPruningResearch")
storage.mount_drive() # Only executes if in Colab

# Automatically resolves to local folder OR /content/drive/MyDrive/...
checkpoint_dir = storage.resolve_path("checkpoints/v1")
```

### Step 2: Preparing the Model (`sp.backends`)
**Why:** You need a unified way to load or build models regardless of whether you use PyTorch or Keras.

```python
from reducnn.backends import get_adapter

# 1. Provide your model object
my_model = ... 

# 2. Get the appropriate adapter (detects Torch vs Keras automatically)
adapter = get_adapter(my_model)

# 3. Load pre-trained weights (Unified syntax)
adapter.load_checkpoint(my_model, checkpoint_dir / "weights.pth")
```

You can also copy checkpoints between Colab Drive and repo workspace:

```python
storage.copy_into_project(
    "/content/drive/MyDrive/activation-based-pruning/my_models/resnet18_pretrained.pth",
    "my_models/resnet18_pretrained.pth",
)
```

### Step 3: Diagnostic Research (`sp.analyzer.MethodValidator`)
**Why:** Before pruning, you may want to see if 'taylor' scores correlate with 'l1_norm' or 'apoz' for your specific architecture.

```python
from reducnn.analyzer import MethodValidator

validator = MethodValidator()
validator.compare_methods(
    model=my_model, 
    loader=my_dataloader, 
    methods=['l1_norm', 'mean_abs_act', 'apoz'],
    ratio=0.3
)
# This generates diagnostic heatmaps and rank-correlation plots.
```

### Step 4: Executing the Pruning (`sp.pruner.ReduCNNPruner`)
**Why:** This is the core engine. It calculates importance and performs "surgery" to return a physically smaller model.

```python
from reducnn.pruner import ReduCNNPruner

surgeon = ReduCNNPruner(method='apoz', scope='local')

# Returns a new physically smaller model, masks, and surgery duration (seconds)
pruned_model, masks, duration = surgeon.prune(
    my_model,
    my_dataloader,
    ratio=0.4,
    save_pruned_path="exports/pruned_model.pth",  # Optional
)
```

For custom pre-trained models:

```python
pruned_model, masks, duration = surgeon.prune_custom_model(
    model=my_model,
    loader=my_dataloader,
    ratio=0.4,
    checkpoint_path="my_models/pretrained_weights.pth",  # Optional load
    save_pruned_path="exports/pruned_model.pth",         # Optional save
)
```

### Step 5: Efficiency Trade-off Analysis (`sp.analyzer.ParetoAnalyzer`)
**Why:** Stakeholders need to know the "ROI"—how much accuracy is lost for every 10% of FLOPs reduced.

```python
from reducnn.analyzer import ParetoAnalyzer

pareto = ParetoAnalyzer(method='apoz')
pareto.run(my_model, my_dataloader, ratios=[0.2, 0.4, 0.6, 0.8])
# Generates the Pareto Frontier curves (Accuracy vs Compute).
```

### Step 6: Presentation Visuals (`sp.visualization`)
**Why:** To prove the pruning was "surgical" and didn't break model internal representations.

```python
import reducnn.visualization as viz

# 1. Bar chart of layer-wise sensitivity
viz.plot_layer_sensitivity(masks, title_prefix="VGG16")

# 2. Compare Params and FLOPs
b_stats = adapter.get_stats(my_model)
p_stats = adapter.get_stats(pruned_model)
viz.plot_metrics_comparison(b_stats, p_stats)
```

---

## High-Level Orchestration (`sp.engine.Orchestrator`)

If you prefer a single command to run the full "Train -> Prune -> Fine-tune" pipeline, use the Orchestrator:

```python
from reducnn.engine import Orchestrator

config = {'backend': 'pytorch', 'model_type': 'vgg16', 'ratio': 0.4, 'method': 'apoz', 'epochs': 10}
orch = Orchestrator(config)
orch.run(train_loader, val_loader=val_loader)
```

---

## Production Defaults (v0.88)

ReduCNN now defaults to production-style behavior:

1. Baseline policy (`load-or-train`)
- Baseline runs auto-load latest checkpoint when available.
- If missing, baseline is trained and auto-saved.
- Paths:
  - `saved_models/baselines/pytorch/<dataset>/<model>/...`
  - `saved_models/baselines/keras/<dataset>/<model>/...`

2. Calibration policy
- If no batch limit is provided, scoring uses full calibration loader length.
- Optional overrides:
  - `prune_batches`
  - `calib_batches`
  - `calibration_batches`

3. Artifact persistence
- Set:
  - `REDUCNN_ARTIFACT_DIR`
  - `REDUCNN_ARTIFACT_MIRROR_DIR` (optional)
  - `REDUCNN_RUN_ID` (optional)
- Visualization outputs are auto-persisted for thesis/presentation workflows.

---

## Interactive Help & Documentation

The package follows professional Python standards. You can access detailed instructions for any class or module directly in your terminal or notebook using `help()`:

```python
import reducnn.pruner as pruner

# Show documentation for the ReduCNNPruner class
help(pruner.ReduCNNPruner)

# Show documentation for the Pruning Engine module
help(pruner)
```

## Advanced: Adding Custom Math

You can register your own pruning criteria using the `@register_method` decorator:

```python
from reducnn.pruner import register_method
import numpy as np

@register_method("my_custom_math")
def my_custom_score(layer, **kwargs):
    # 'layer' is automatically passed by the adapter
    weights = layer.get_weights()[0] # Keras example
    return np.mean(np.abs(weights), axis=(0,1,2))

# Now you can use it in the surgeon
surgeon = ReduCNNPruner(method="my_custom_math")
```
