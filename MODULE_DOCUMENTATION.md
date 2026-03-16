# Module Documentation: `surgical_pruning` 📦

This document provides a comprehensive breakdown of every module, class, and function within the `surgical_pruning` package. It is designed for developers and researchers who need to understand the internal "plumbing" of the library.

---

## 1. `surgical_pruning.core`
The foundational layer of the library, containing abstract interfaces, decorators, and infrastructure utilities.

### `adapter.py`
- **Purpose:** Defines the `FrameworkAdapter` Abstract Base Class (ABC).
- **Key Class:** `FrameworkAdapter`
    - `get_model(model_type)`: Abstract method to initialize standard architectures.
    - `train(...)`: Abstract method for model optimization.
    - `evaluate(...)`: Abstract method for accuracy calculation.
    - `get_score_map(...)`: Abstract method to calculate importance scores for every filter.
    - `apply_surgery(...)`: The entry point for physical weight deletion.
    - `load_checkpoint() / save_checkpoint()`: Unified weight I/O.

### `decorators.py`
- **Purpose:** Enhances the developer experience and implements the "plug-and-play" backend logic.
- **Key Functions:**
    - `@framework_dispatch`: **The Core Innovation.** Automatically detects if a model is PyTorch or Keras and injects the correct adapter into the function.
    - `@timer`: Measures performance overhead of pruning/training.
    - `@logger`: Standardizes the visual logging of the pipeline phases.
    - `get_framework_adapter()`: Logic for dynamic backend discovery based on Python object types.

### `storage.py`
- **Purpose:** Standardizes model saving/loading between Local VS Code and Google Colab.
- **Key Class:** `CloudStorage`
    - `mount_drive()`: Handles Google Drive handshakes.
    - `resolve_path(rel_path)`: Dynamically routes paths. Prevents "File Not Found" errors when moving from a laptop to a cloud GPU.

### `exceptions.py`
- **Purpose:** Defines custom errors for the package.
- **Exceptions:** `SurgeryError`, `UnsupportedFrameworkError`, `MethodRegistrationError`.

---

## 2. `surgical_pruning.backends`
Contains the concrete implementations of the `FrameworkAdapter` for specific deep learning engines.

### `torch_backend.py`
- **Purpose:** Implements the `PyTorchAdapter` and the logic for dynamic graph surgery.
- **Key Class:** `TorchStructuralPruner`
    - `_trace()`: Recursively walks the PyTorch module tree to find Conv-BN-Conv dependency chains.
    - `_shrink()`: The function that actually deletes rows/columns from `nn.Parameter` tensors.
- **Key Class:** `PyTorchAdapter`
    - Handles Taylor pruning using `register_full_backward_hook`.
    - Manages GPU/CPU device mapping automatically.

### `keras_backend.py`
- **Purpose:** Implements the `KerasAdapter` and the functional rebuilder for static graphs.
- **Key Class:** `KerasAdapter`
    - `_apply_structural_pruning()`: A recursive rebuilder that creates a brand new `Model` by copying and slicing weights from the original.
    - `_estimate_flops()`: An analytical FLOPs counter that traverses the Keras layer list.
    - Handles Taylor pruning using `tf.GradientTape`.

---

## 3. `surgical_pruning.pruner`
The "Brain" of the library. It contains the mathematical heuristics and the logic for decision-making.

### `surgeon.py`
- **Purpose:** Orchestrates the pruning action.
- **Key Class:** `SurgicalPruner`
    - `prune(model, loader, ratio)`: Coordinates the 3-step process: (1) Score Calculation, (2) Mask Selection, (3) Structural Surgery.

### `registry.py`
- **Purpose:** A central hub for importance heuristics.
- **Key Function:** `@register_method(name)`: A decorator that allows researchers to "drop in" new pruning math without editing the core library.

### `criteria.py`
- **Purpose:** Implements the built-in pruning algorithms (L1, L2, Taylor, Random).
- **Key Function:** `taylor_score()`: Native implementation of gradient-based filter importance.

### `mask_builder.py`
- **Purpose:** The logic for selecting which filters to keep based on the calculated scores.
- **Key Function:** `build_pruning_masks()`: Implements both **Local** (layer-wise) and **Global** (network-wide) thresholding strategies.

---

## 4. `surgical_pruning.analyzer`
Diagnostic tools for research and decision-making.

### `validator.py`
- **Purpose:** Compares different math heuristics.
- **Key Method:** `compare_methods()`: Calculates raw scores for multiple methods (e.g., L1 vs Taylor) and uses Spearman correlations to check their agreement.

### `pareto.py`
- **Purpose:** ROI analysis for stakeholders.
- **Key Method:** `run()`: Executes a "Stress Test" by pruning the model at multiple intensities (20%, 40%, 60%, 80%) to generate the Accuracy vs. Efficiency tradeoff curve.

---

## 5. `surgical_pruning.visualization`
Decoupled plotting utilities for both technical research and business presentations.

### `stakeholder.py`
- **Purpose:** "Big Picture" visuals.
- **Functions:** 
    - `plot_layer_sensitivity()`: A Red-to-Green bar chart showing which parts of the brain were removed.
    - `plot_metrics_comparison()`: Side-by-side Params/FLOPs comparison.
    - `plot_training_history()`: Visualizes fine-tuning convergence.

### `research.py`
- **Purpose:** Deep-dive diagnostics.
- **Functions:**
    - `plot_score_distributions()`: Histogram of filter importance.
    - `plot_rank_correlation()`: Heatmap of method agreement.

---

## 6. `surgical_pruning.engine`
The "High-Level" entry point for standard workflows.

### `orchestrator.py`
- **Purpose:** A thin wrapper for users who want to run the full pipeline with a single command.
- **Key Class:** `Orchestrator`
    - `run()`: Trains a baseline, visualizes it, prunes it, fine-tunes the result, and generates a final report.
