# Integrated VS Code + Google Colab Workflow

This document outlines the professional "surgical" workflow for developing the `surgical_pruning` package locally in **VS Code** while executing experiments on **Google Colab**'s GPU/TPU infrastructure.

## 1. Prerequisites
- **Google Drive for Desktop:** Installed on your local machine.
- **Project Sync:** Your `activation-based-pruning` folder must be synced to Google Drive (e.g., via the "Other computers" or "My Drive" feature).
- **VS Code:** Open the local project folder and perform your edits there.

## 2. Colab Initialization (The "Bootloader")
Run this entire block in a single Colab cell at the start of your session.

```python
# --- 1. FIX FOR PYTHON 3.12 'imp' MISSING ---
import sys
import types
import importlib
if 'imp' not in sys.modules:
    imp = types.ModuleType('imp')
    imp.reload = importlib.reload
    sys.modules['imp'] = imp

from google.colab import drive
import os

# --- 2. MOUNT GOOGLE DRIVE ---
drive.mount('/content/drive')

# --- 3. FIND AND NAVIGATE TO PROJECT ROOT ---
# This search handles "Other computers" vs "My Drive" automatically
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
    
    # --- 4. THE "SURGICAL" FIX FOR THE SRC LAYOUT ---
    # Add 'src' to path so 'import surgical_pruning' works directly
    src_path = os.path.join(project_path, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # --- 5. SURGICAL EDITABLE INSTALL ---
    !pip install -e .
    
    # --- 6. ENABLE AUTO-RELOAD MAGIC ---
    %load_ext autoreload
    %autoreload 2
    
    print(f"✅ Workflow Ready: {project_path}")
    !ls -F
else:
    print(f"❌ Error: Could not find '{project_folder_name}' on Google Drive.")
```

## 3. Daily "Seamless" Cycle
1. **Local Edit:** Modify your code in `src/surgical_pruning/` using VS Code.
2. **Save:** Hit `Ctrl+S`. Google Drive for Desktop will sync the file to the cloud in 1-2 seconds.
3. **Colab Run:** Simply run your notebook cell in Colab. 
   - **No re-installing.** 
   - **No restarting the kernel.**
   - The `%autoreload` magic detects the change on Drive and automatically updates the module in your current session.

## 4. Usage in Colab
Once the bootloader has run, you can import your package directly:

```python
from surgical_pruning.engine.orchestrator import PruningOrchestrator
orchestrator = PruningOrchestrator(framework="torch")
print("Module imported successfully!")
```

## 5. Troubleshooting
- **Sync Lag:** If a change isn't reflected, wait 5 seconds for Google Drive to finish the upload.
- **Module Not Found:** Ensure you have a `pyproject.toml` in the project root with the `[tool.setuptools.packages.find] where = ["src"]` configuration.
- **Permission Denied:** Ensure you granted Colab access to your Google Drive when the mounting popup appeared.
