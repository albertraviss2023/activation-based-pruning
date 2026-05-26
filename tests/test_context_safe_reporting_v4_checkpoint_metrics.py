import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "report_artifacts" / "context_safe_hybrid_singular_report_v4_model_metrics"
TABLE_DIR = REPORT_DIR / "tables"
NOTEBOOK_PATH = PROJECT_ROOT / "context_safe_hybrid_singular_reporting_v4_model_metrics.ipynb"


def _read_required_table(name: str) -> pd.DataFrame:
    path = TABLE_DIR / name
    assert path.exists() and path.stat().st_size > 0, f"missing v4 table: {path}"
    return pd.read_csv(path)


def test_v4_checkpoint_direct_metrics_exist_and_include_direct_flops():
    metrics = _read_required_table("v4_checkpoint_direct_model_metrics.csv")
    comparison = _read_required_table("v4_hybrid_vs_singular_checkpoint_direct_long.csv")

    for col in [
        "record_type",
        "metric_status",
        "checkpoint_path_resolved",
        "direct_baseline_flops",
        "direct_model_flops",
        "direct_flops_reduction_pct",
        "direct_baseline_params",
        "direct_model_params",
        "direct_params_reduction_pct",
    ]:
        assert col in metrics.columns, f"v4 direct metric table missing {col}"

    ok = metrics[metrics["metric_status"].astype(str).eq("ok")].copy()
    assert not ok.empty, "v4 did not successfully profile any saved checkpoints"
    assert ok["direct_flops_reduction_pct"].notna().any(), "v4 has no checkpoint-derived FLOPs reductions"
    assert ok["direct_params_reduction_pct"].notna().any(), "v4 has no checkpoint-derived parameter reductions"

    direct_rows = comparison[comparison["metric"].astype(str).eq("direct_flops_reduction_pct")]
    assert not direct_rows.empty, "v4 comparison table has no direct FLOPs rows"
    assert direct_rows["metric_source"].astype(str).eq("checkpoint_direct").all()


def test_v4_plots_and_layerwise_direct_tables_are_present():
    manifest = _read_required_table("v4_plot_manifest_checkpoint_comparisons.csv")
    layerwise = _read_required_table("v4_hybrid_layerwise_policy_linked_to_direct_metrics.csv")

    assert "comparison_plot" in manifest.columns
    plotted = manifest[manifest["comparison_plot"].astype(str).str.len().gt(0)].copy()
    assert not plotted.empty, "v4 did not create checkpoint-derived comparison plots"
    for value in plotted["comparison_plot"].head(20):
        path = Path(str(value))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        assert path.exists(), f"v4 plot listed in manifest is missing: {path}"
        assert path.stat().st_size > 1024, f"v4 plot is unexpectedly small: {path}"

    assert "report_stack_id" in layerwise.columns
    assert "direct_flops_reduction_pct" in layerwise.columns
    assert not layerwise.empty, "v4 layerwise policy linkage table is empty"


def test_v4_notebook_runs_builder_and_displays_comparison_gallery():
    assert NOTEBOOK_PATH.exists(), f"missing v4 reporting notebook: {NOTEBOOK_PATH}"
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))

    assert "build_context_safe_report_v4_from_checkpoints.py" in source
    assert "V4 checkpoint-derived comparison plots" in source
    assert "v4_hybrid_layerwise_policy_linked_to_direct_metrics.csv" in source
    assert "Image(filename=str(path))" in source
