# ReduCNN Experiment Guide

This guide provides an overview of the experiment notebooks included in the ReduCNN repository and instructions for running them.

## 1. Experiment Overview

The notebooks are organized by their primary research objective:

### Pruning Policy Search (LFPC)
These notebooks discover and benchmark layer-wise hybrid pruning stacks.
- **Naming Pattern**: `experiments_for_pruning_policy_search_on_context_<dataset>_<model>_...`
- **Goal**: Find the optimal combination of pruning methods across layers to maximize accuracy while minimizing FLOPs or inference time.

### Dataset-Specific Benchmarks
- `experiments_cifar10.ipynb`: Standard CIFAR-10 pruning benchmarks.
* `experiments_cat_dog.ipynb`: Binary classification pruning on the Cats-vs-Dogs dataset.

### Specialized Reporting & Reproduction
- `singular_method_context_pruning_report.ipynb`: Generates comparative reports for single-method pruning.
- `example of reproducing_discovered_pruning policies.ipynb`: Demonstrates how to load and verify a discovered hybrid stack.

## 2. Running Experiments

### Option A: VS Code (Local)
1. **Open Workspace**: Open the repository folder in VS Code.
2. **Select Kernel**: Open a `.ipynb` file and ensure your Python environment (with ReduCNN installed) is selected as the Jupyter kernel.
3. **Run Cells**: Execute the cells sequentially. Artifacts (checkpoints, plots) will be saved to `saved_models/` and `outputs/`.

### Option B: Google Colab
1. **Upload/Clone**: Open a new notebook in Colab and clone the repo:
   ```python
   !git clone https://github.com/albertraviss2023/activation-based-pruning.git
   %cd activation-based-pruning
   !pip install -e .
   ```
2. **GPU Support**: Ensure the Colab runtime is set to **GPU** (Runtime -> Change runtime type).
3. **Run Experiments**: Open any of the project notebooks via Colab's file browser and run the cells.

## 3. Results and Artifacts
Every run generates a unique `RUN_ID`. 
- **Tables & Plots**: Found in `outputs/experiments/<dataset>/<model>/<run_id>/`.
- **Model Checkpoints**: Found in `saved_models/fine_tuned/`.

Refer to [docs/WORKFLOWS_HOWTO.md](WORKFLOWS_HOWTO.md) for deeper workflow details.
