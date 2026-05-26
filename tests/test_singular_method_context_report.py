import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "report_artifacts" / "singular_method_context_report"
TABLE_DIR = REPORT_DIR / "tables"
NOTEBOOK_PATH = PROJECT_ROOT / "singular_method_context_pruning_report.ipynb"


def _read_required_table(name: str) -> pd.DataFrame:
    path = TABLE_DIR / name
    assert path.exists() and path.stat().st_size > 0, f"missing singular report table: {path}"
    return pd.read_csv(path)


def test_singular_context_report_has_direct_structural_metrics_and_plots():
    metrics = _read_required_table("singular_method_context_metrics.csv")
    plots = _read_required_table("plot_manifest_singular_method_context_metrics.csv")

    for col in [
        "dataset",
        "model",
        "scope",
        "ratio",
        "method",
        "accuracy_delta_pp",
        "direct_flops_reduction_pct",
        "direct_params_reduction_pct",
        "metric_status",
    ]:
        assert col in metrics.columns, f"singular metrics missing {col}"

    ok = metrics[metrics["metric_status"].astype(str).eq("ok")]
    assert not ok.empty, "no singular checkpoints were profiled successfully"
    assert ok["direct_flops_reduction_pct"].notna().any()
    assert ok["direct_params_reduction_pct"].notna().any()

    assert not plots.empty, "no singular context plots generated"
    for col in ["combined_plot", "accuracy_delta_plot", "flops_reduction_plot", "params_reduction_plot"]:
        assert col in plots.columns
        sample = plots[col].dropna().astype(str).head(10)
        assert not sample.empty
        for value in sample:
            path = Path(value)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            assert path.exists(), f"plot listed in manifest is missing: {path}"
            assert path.stat().st_size > 1024, f"plot is unexpectedly small: {path}"


def test_singular_context_notebook_displays_requested_three_metric_report():
    assert NOTEBOOK_PATH.exists(), f"missing singular report notebook: {NOTEBOOK_PATH}"
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))

    assert "build_singular_method_context_report.py" in source
    assert "accuracy_delta_pp" in source
    assert "direct_flops_reduction_pct" in source
    assert "direct_params_reduction_pct" in source
    assert "Singular method metric plots by exact context" in source
