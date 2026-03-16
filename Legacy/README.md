# Surgical Pruning ✂️

A professional, modular, dual-framework (PyTorch & Keras) Python package for Activation-Based Structural Pruning.

## Features

- **Dual-Framework:** Seamlessly pass PyTorch (`nn.Module`) or Keras (`keras.Model`) objects. The `@framework_dispatch` decorator handles the rest.
- **Highly Composable:** Use only the parts you need. Prune without plotting, or plot without pruning.
- **Physical Surgery:** True structural pruning (not just masking). The resulting models use fewer FLOPs and less memory.
- **Extensible:** Register your own custom pruning math with a simple decorator.

## Installation

```bash
pip install -e .
```

## Quick Start (Decoupled Workflow)

Instead of a monolithic pipeline, `surgical_pruning` is designed for on-demand use:

```python
import surgical_pruning.pruner as pruner
import surgical_pruning.visualization as viz

# 1. Provide your model and data (PyTorch or Keras)
my_model = ... # Your trained VGG16 model
my_dataloader = ... 

# 2. Initialize the Pruner
surgeon = pruner.SurgicalPruner(method='taylor', scope='local')

# 3. Prune the model (Returns a smaller, structurally pruned model)
pruned_model, masks = surgeon.prune(my_model, my_dataloader, ratio=0.4)

# 4. (Optional) Visualize for Stakeholders
viz.plot_layer_sensitivity(masks, title_prefix="My Pruned Model")
```

## Independent Troubleshooting

Every visualization and analyzer module can be run directly from the command line with dummy data to verify the plots without needing to load deep learning frameworks:

```bash
python src/surgical_pruning/visualization/stakeholder.py
```
