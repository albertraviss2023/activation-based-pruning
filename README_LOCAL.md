# Local Development Manual: `reducnn` 🛠️

This document is for internal development within VS Code or Colab. It explains how to maintain, troubleshoot, and extend the package before final release.

## 1. Development Environment Setup

To ensure you can import the package while editing the source code in `src/`, set your `PYTHONPATH` or install in editable mode.

### Option A: Editable Install (Recommended)
This allows you to change code in `src/` and have it reflect immediately in your scripts/notebooks.
```bash
pip install -e .
```

---

## 2. Independent Module Troubleshooting

One of the core design goals is that **modules should be testable without the full pipeline.** Many files contain `if __name__ == "__main__":` blocks with mock data.

### Test Visuals
Run this to verify bar charts and metric plots:
```bash
python src/reducnn/visualization/stakeholder.py
```

---

## 3. Verifying Changes (Testing Suite)

Always run the full suite after modifying the core engine or backends.

```bash
# Run integration tests (requires torch and tensorflow)
python -m pytest tests/test_package_integration.py
```

---

## 4. Package Structure Breakdown

| Directory | Purpose |
| :--- | :--- |
| `src/reducnn/core` | ABCs, Decorators (`@framework_dispatch`), and `CloudStorage`. |
| `src/reducnn/backends` | PyTorch/Keras specific logic (Surgery, Stats, Loading). |
| `src/reducnn/pruner` | Core math: `ReduCNNPruner`, `registry`, and `mask_builder`. |
| `src/reducnn/analyzer` | Diagnostic tools: `MethodValidator` and `ParetoAnalyzer`. |
| `src/reducnn/visualization` | All plotting logic decoupled from frameworks. |
| `src/reducnn/engine` | The high-level `Orchestrator` wrapper. |

---

## 5. Integrated VS Code + Google Colab Workflow

This is the recommended workflow for using Colab's GPU/TPU while writing code in VS Code.

### Step 1: The Colab "Bootloader"
Run this entire cell at the start of every Colab session to mount your Drive, link your project, and perform a **surgical editable install**.

```python
# --- 1. FIX FOR PYTHON 3.12 'imp' MISSING ---
import sys, types, importlib
if 'imp' not in sys.modules:
    imp = types.ModuleType('imp'); imp.reload = importlib.reload; sys.modules['imp'] = imp

from google.colab import drive
import os

# --- 2. MOUNT & NAVIGATE ---
drive.mount('/content/drive')
project_folder_name = "activation-based-pruning"
base_paths = ["/content/drive/Othercomputers", "/content/drive/MyDrive"]
project_path = None

for base in base_paths:
    if os.path.exists(base):
        for root, dirs, files in os.walk(base):
            if project_folder_name in dirs:
                project_path = os.path.join(root, project_folder_name)
                break
    if project_path: break

if project_path:
    os.chdir(project_path)
    # --- 3. SURGICAL EDITABLE INSTALL ---
    !pip install -e .
    %load_ext autoreload
    %autoreload 2
    if project_path not in sys.path: sys.path.append(project_path)
    print(f"✅ Workflow Ready: {project_path}")
else:
    print(f"❌ Error: Could not find '{project_folder_name}' on Google Drive.")
```

### Step 2: Research & Experimentation
Now you can import and use the package. Any change you save in VS Code will sync to Google Drive and be automatically reloaded by Colab's `%autoreload` magic.

```python
import reducnn as sp
from reducnn.backends.torch_backend import PyTorchAdapter
from reducnn.pruner import ReduCNNPruner
from reducnn.core.storage import CloudStorage

# --- Storage Configuration (Saves to Project Root) ---
storage = CloudStorage()
checkpoint_dir = storage.resolve_path("my_models/checkpoints")

# --- Initialize Backend ---
adapter = PyTorchAdapter(config={'lr': 3e-4})
model = adapter.get_model("vgg16")

# --- Pruning Execution ---
# (Assumes 'train_loader' is defined)
surgeon = ReduCNNPruner(method='taylor', ratio=0.4)
pruned_model, masks = surgeon.prune(model, train_loader)

# --- Save to Google Drive ---
adapter.save_checkpoint(pruned_model, checkpoint_dir / "vgg16_pruned_40.pth")
print(f"✅ Pruned model saved to: {checkpoint_dir}")
```

### Why this is the "Pro" way:
*   **Persistent Storage:** Models are saved directly back to your local machine via Google Drive sync.
*   **Seamless Edits:** Hit `Ctrl+S` in VS Code, wait 2 seconds, and re-run your Colab cell.
*   **No Clutter:** Only the `src/` folder is installed. Legacy notebooks and data files are ignored by Python.
