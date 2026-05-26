from pathlib import Path

import pandas as pd

from scripts.build_policy_parameter_sensitivity_report import build_report


def test_policy_parameter_sensitivity_report_builds(tmp_path):
    out_dir = tmp_path / "policy_sensitivity"
    result = build_report(
        Path("reports/experiment_registry/contexts.csv"),
        out_dir,
        Path("report_artifacts/context_safe_hybrid_singular_report_v4_model_metrics/tables/v4_checkpoint_direct_model_metrics.csv"),
        max_layer_heatmaps=2,
        write_analysis_notebook=False,
    )

    assert result.hybrid_rows > 0
    assert result.policy_rows > 0
    assert result.context_rows > 0
    assert result.layer_rows > 0
    assert result.plots >= 5

    summary = pd.read_csv(out_dir / "tables" / "policy_stability_by_context.csv")
    layer = pd.read_csv(out_dir / "tables" / "layerwise_policy_stability.csv")
    response = pd.read_csv(out_dir / "tables" / "threshold_response_by_context.csv")
    manifest = pd.read_csv(out_dir / "tables" / "plot_manifest.csv")

    required_summary_cols = {
        "objective_label",
        "dataset",
        "model",
        "scope",
        "ratio",
        "num_threshold_settings",
        "modal_policy_share",
        "mean_layer_dominant_share",
        "accuracy_delta_range",
        "flops_reduction_range",
        "time_sec_range",
        "stability_label",
    }
    assert required_summary_cols.issubset(summary.columns)
    assert not summary["modal_policy_share"].isna().all()
    assert summary["modal_policy_share"].between(0, 1).all()

    required_layer_cols = {
        "objective_label",
        "dataset",
        "model",
        "scope",
        "ratio",
        "layer",
        "dominant_method_display",
        "dominant_method_share",
        "method_frequency_json",
    }
    assert required_layer_cols.issubset(layer.columns)
    assert layer["dominant_method_share"].between(0, 1).all()

    assert {"variance_threshold", "spearman_threshold", "jaccard_threshold"}.issubset(response.columns)
    assert not manifest.empty
    for plot_path in manifest["plot"].head(5):
        assert Path(plot_path).exists()


def test_policy_parameter_sensitivity_notebook_exists():
    assert Path("policy_parameter_sensitivity_analysis_v2.ipynb").exists()
