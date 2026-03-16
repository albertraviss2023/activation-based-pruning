# Surgical Pruning ✂️

A professional, modular, dual-framework Python package for **Activation-Based Structural Pruning**. This package allows you to physically remove filters and channels from PyTorch and Keras models, reducing compute cost (FLOPs) and memory footprint while maintaining behavioral consistency.

## Installation

To install the package in editable mode (perfect for research and development):

```bash
git clone https://github.com/your-repo/activation-based-pruning.git
cd activation-based-pruning
pip install -e .
```

---

## Sequential Decoupled Workflow

Unlike a monolithic script, `surgical_pruning` is designed as a suite of independent tools. Here is how to use them sequentially:

### Step 1: Environment & Storage Setup (`sp.CloudStorage`)
**Why:** Researchers often switch between local VS Code and Google Colab. This module ensures your model paths remain consistent without changing code.

```python
import surgical_pruning as sp

# project_name maps to a folder in your Google Drive 'MyDrive'
storage = sp.CloudStorage(project_name="MyPruningResearch")
storage.mount_drive() # Only executes if in Colab

# Automatically resolves to local folder OR /content/drive/MyDrive/...
checkpoint_dir = storage.resolve_path("checkpoints/v1")
```

### Step 2: Preparing the Model (`sp.backends`)
**Why:** You need a unified way to load or build models regardless of whether you use PyTorch or Keras.

```python
from surgical_pruning.backends import get_adapter

# 1. Provide your model object
my_model = ... 

# 2. Get the appropriate adapter (detects Torch vs Keras automatically)
adapter = get_adapter(my_model)

# 3. Load pre-trained weights (Unified syntax)
adapter.load_checkpoint(my_model, checkpoint_dir / "weights.pth")
```

### Step 3: Diagnostic Research (`sp.analyzer.MethodValidator`)
**Why:** Before pruning, you may want to see if 'taylor' scores correlate with 'l1_norm' or 'apoz' for your specific architecture.

```python
from surgical_pruning.analyzer import MethodValidator

validator = MethodValidator()
validator.compare_methods(
    model=my_model, 
    loader=my_dataloader, 
    methods=['l1_norm', 'taylor', 'apoz'],
    ratio=0.3
)
# This generates diagnostic heatmaps and rank-correlation plots.
```

### Step 4: Executing the Pruning (`sp.pruner.SurgicalPruner`)
**Why:** This is the core engine. It calculates importance and performs "surgery" to return a physically smaller model.

```python
from surgical_pruning.pruner import SurgicalPruner

surgeon = SurgicalPruner(method='taylor', scope='local')

# Returns a new physically smaller model and the boolean masks used
pruned_model, masks = surgeon.prune(my_model, my_dataloader, ratio=0.4)
```

### Step 5: Efficiency Trade-off Analysis (`sp.analyzer.ParetoAnalyzer`)
**Why:** Stakeholders need to know the "ROI"—how much accuracy is lost for every 10% of FLOPs reduced.

```python
from surgical_pruning.analyzer import ParetoAnalyzer

pareto = ParetoAnalyzer(method='taylor')
pareto.run(my_model, my_dataloader, ratios=[0.2, 0.4, 0.6, 0.8])
# Generates the Pareto Frontier curves (Accuracy vs Compute).
```

### Step 6: Presentation Visuals (`sp.visualization`)
**Why:** To prove the pruning was "surgical" and didn't break model internal representations.

```python
import surgical_pruning.visualization as viz

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
from surgical_pruning.engine import Orchestrator

config = {'model_type': 'vgg16', 'ratio': 0.4, 'method': 'taylor', 'epochs': 10}
orch = Orchestrator(config)
orch.run(train_loader, val_loader=val_loader)
```

---

## Interactive Help & Documentation

The package follows professional Python standards. You can access detailed instructions for any class or module directly in your terminal or notebook using `help()`:

```python
import surgical_pruning.pruner as pruner

# Show documentation for the SurgicalPruner class
help(pruner.SurgicalPruner)

# Show documentation for the Pruning Engine module
help(pruner)
```

## Advanced: Adding Custom Math

You can register your own pruning criteria using the `@register_method` decorator:

```python
from surgical_pruning.pruner import register_method
import numpy as np

@register_method("my_custom_math")
def my_custom_score(layer, **kwargs):
    # 'layer' is automatically passed by the adapter
    weights = layer.get_weights()[0] # Keras example
    return np.mean(np.abs(weights), axis=(0,1,2))

# Now you can use it in the surgeon
surgeon = SurgicalPruner(method="my_custom_math")
```
