from __future__ import annotations

import json
from pathlib import Path


def _find_cell_source(nb, needle: str) -> str:
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if needle in src:
            return src
    raise AssertionError(f"Could not find notebook cell containing: {needle}")


def test_deep_dive_notebook_has_backend_conditioned_imports_and_loaders():
    nb_path = Path("visualization_deep_dive.ipynb")
    payload = json.loads(nb_path.read_text(encoding="utf-8"))

    imports_cell = _find_cell_source(payload, "from reducnn.backends.factory import get_adapter")
    data_cell = _find_cell_source(payload, "if BACKEND == 'pytorch':")
    policy_cell = _find_cell_source(payload, "DEMO_MODE =")
    model_cell = _find_cell_source(payload, "# Backend-agnostic model loading with candidate fallback")

    # Imports are now backend-agnostic in the core import cell.
    assert "import torch" not in imports_cell
    assert "import torchvision" not in imports_cell

    # Data loading has explicit backend branches.
    assert "if BACKEND == 'pytorch':" in data_cell
    assert "import torchvision" in data_cell
    assert "import tensorflow as tf" in data_cell

    # Dedicated policy controls exist.
    assert "DEMO_MODE" in policy_cell
    assert "CHECKPOINT_STAMP" in policy_cell
    assert "baselines" in policy_cell
    assert "pruned_raw" in policy_cell
    assert "fine_tuned" in policy_cell
    assert "legacy_baseline_name" in policy_cell
    assert "build_model_paths" in policy_cell

    # Model loading/checkpoint path supports backend-specific semantics.
    assert "pretrained=USE_PRETRAINED_INIT" in model_cell
    assert "MODEL_PATHS = build_model_paths" in model_cell
