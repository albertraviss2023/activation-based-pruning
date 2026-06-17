import json
from pathlib import Path


def test_resnet_catdog_flops_accuracy_notebook_rejects_empty_flops_exports():
    notebook = Path(
        "experiments_for_pruning_policy_search_on_context_cats_dogs_catdog_baseline_resnet18_registered_methods_objective_flops_accuracy.ipynb"
    )
    assert notebook.exists()
    data = json.loads(notebook.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in data.get("cells", []))

    assert "_notebook_hook_flops" in source
    assert "Adapter FLOPs stats failed" in source
    assert "_validate_flops_export_frame" in source
    assert "f\"{artifact_name}_flops_export_audit.csv\"" in source
    assert "FLOPs export audit failed for {artifact_name}" in source
