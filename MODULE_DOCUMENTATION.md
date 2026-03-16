# Module Documentation: `reducnn` 📦

This document provides a comprehensive breakdown of every module, class, and function within the `reducnn` package. It is designed for developers and researchers who need to understand the internal "plumbing" of the library.

---

## 1. `reducnn.core`
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

## 2. `reducnn.backends`
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

## 3. `reducnn.pruner`
The "Brain" of the library. It contains the mathematical heuristics and the logic for decision-making.

### `surgeon.py`
- **Purpose:** Orchestrates the pruning action.
- **Key Class:** `ReduCNNPruner`
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

## 4. `reducnn.analyzer`
Diagnostic tools for research and decision-making.

### `validator.py`
- **Purpose:** Compares different math heuristics.
- **Key Method:** `compare_methods()`: Calculates raw scores for multiple methods (e.g., L1 vs Taylor) and uses Spearman correlations to check their agreement.

### `pareto.py`
- **Purpose:** ROI analysis for stakeholders.
- **Key Method:** `run()`: Executes a "Stress Test" by pruning the model at multiple intensities (20%, 40%, 60%, 80%) to generate the Accuracy vs. Efficiency tradeoff curve.

---

## 5. `reducnn.visualization`
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

## 6. Dataset & Model Generalization
The framework is designed to be dataset-agnostic, allowing researchers to swap between standard benchmarks and custom datasets with zero changes to the core engine.

### Dataset Auto-Discovery
- **Shape Inference:** The `FrameworkAdapter` (both PyTorch and Keras) automatically detects the `input_shape` (e.g., 3x32x32 for CIFAR, 3x128x128 for CatDog) from the provided calibration data loader.
- **Classification Head Adaptation:** The `get_model` factory dynamically adjusts the final `Dense` or `Linear` layer to match the `num_classes` parameter (e.g., 2 for CatDog, 10 for CIFAR-10, 100 for CIFAR-100).
- **Normalization Handling:** Built-in training loops support any standard `DataLoader` or `tf.data.Dataset` object, ensuring that custom pre-processing and augmentations are preserved during the "healing" (fine-tuning) phase.

### Architectural Surgery: VGG vs. ResNet
The framework distinguishes between **Linear Dependency Chains** and **Residual Clusters**:

#### 1. VGG-Style (Sequential)
- **Pattern:** `Conv -> BN -> ReLU -> Conv`
- **Logic:** A "Cascading Cut" is performed. Removing filter $j$ in the first Conv requires removing index $j$ from the following BN and removing index $j$ from the **input channels** of the next Conv.
- **Tracing:** Simple sequential look-ahead is sufficient to identify the next prunable layer.

#### 2. ResNet-Style (Branched/Residual)
- **Pattern:** `(Identity + Conv_Block) -> Add`
- **Problem:** Because the output of the identity shortcut and the residual block are added, they MUST have the same number of channels. Independent pruning would cause a shape mismatch error.
- **Solution (Cluster Harmonization):** 
    - **Tracer:** The `TorchStructuralPruner` uses `torch.fx` to identify `Add` nodes and traces back to all contributing producers.
    - **Harmonizer:** These producers are grouped into a "Cluster." The framework ensures that all members of a cluster receive the **exact same pruning mask**, maintaining mathematical consistency for the element-wise addition.
    - **Idempotency:** The surgery engine tracks which inputs have already been shrunk to prevent "double-slicing" in complex multi-branch graphs.
 **### Architecture & Dataset Generalization
The framework has been upgraded to support non-sequential models and generic datasets:
- **Residual Cluster Management:** (PyTorch) Automatically identifies layers that must be pruned identically due to skip connections (e.g., in ResNet).
- **Functional Rebuilder:** (Keras) Uses a graph-based reconstruction strategy to maintain connectivity in complex branching models.
- **Dynamic Shape Detection:** Adapters now derive input dimensions and class counts automatically from the data loader or model configuration, allowing seamless transitions between MNIST (28x28), CIFAR (32x32), and ImageNet (224x224).
