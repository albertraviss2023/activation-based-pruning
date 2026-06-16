import json
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "report_artifacts" / "context_safe_hybrid_singular_report"
TABLE_DIR = REPORT_DIR / "tables"
NOTEBOOK_PATH = PROJECT_ROOT / "context_safe_hybrid_singular_reporting_v2_memory_safe.ipynb"


def _require_report_table(name: str) -> pd.DataFrame:
    path = TABLE_DIR / name
    if not path.exists() or path.stat().st_size == 0:
        pytest.fail(f"Required context-safe report table is missing or empty: {path}")
    return pd.read_csv(path)


def test_top_hybrid_stacks_have_thesis_comparison_plots():
    manifest = _require_report_table("plot_manifest_layerwise_and_comparison.csv")
    top = _require_report_table("top_hybrid_stacks_by_context.csv")

    assert "comparison_plot" in manifest.columns, "plot manifest lost the comparison_plot column"
    plotted = manifest[manifest["comparison_plot"].astype(str).str.len().gt(0)].copy()
    assert not plotted.empty, "no hybrid-vs-singular comparison plots are listed in the manifest"

    merge_keys = [
        "objective",
        "dataset",
        "model",
        "scope",
        "ratio",
        "context_rank",
        "report_stack_id",
    ]
    required = top[[c for c in merge_keys if c in top.columns]].drop_duplicates()
    available = plotted[[c for c in merge_keys if c in plotted.columns]].drop_duplicates()
    merged = required.merge(available, on=[c for c in merge_keys if c in required.columns and c in available.columns], how="left", indicator=True)
    missing = merged[merged["_merge"].ne("both")]
    assert missing.empty, "top hybrid stacks are missing thesis comparison plots:\n" + missing.to_string(index=False)

    missing_files = []
    invalid_pngs = []
    for value in plotted["comparison_plot"]:
        path = Path(str(value))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            missing_files.append(str(path))
            continue
        if path.stat().st_size < 1024:
            invalid_pngs.append(f"{path} is too small ({path.stat().st_size} bytes)")
            continue
        with path.open("rb") as handle:
            if handle.read(8) != b"\x89PNG\r\n\x1a\n":
                invalid_pngs.append(f"{path} is not a PNG")

    assert not missing_files, "comparison plot files listed in the manifest are missing:\n" + "\n".join(missing_files[:20])
    assert not invalid_pngs, "comparison plot files are invalid:\n" + "\n".join(invalid_pngs[:20])


def test_absolute_scale_and_appendix_plots_are_linked_to_exact_contexts():
    manifest = _require_report_table("plot_manifest_layerwise_and_comparison.csv")
    comparison = _require_report_table("hybrid_vs_singular_exact_context_long.csv")
    baseline = _require_report_table("baseline_model_scale.csv")

    assert len(baseline) == 4
    assert baseline["baseline_gops_invariant"].astype(bool).all()
    assert baseline["baseline_params_invariant"].astype(bool).all()
    for col in [
        "hybrid_model_gops",
        "singular_model_gops",
        "hybrid_model_params_m",
        "singular_model_params_m",
        "operation_count_convention",
    ]:
        assert col in comparison.columns
    assert comparison["hybrid_model_gops"].notna().any()
    assert comparison["singular_model_gops"].notna().any()

    assert "absolute_footprint_plot" in manifest.columns
    plotted = manifest[manifest["absolute_footprint_plot"].fillna("").astype(str).str.len().gt(0)]
    assert not plotted.empty
    for value in plotted["absolute_footprint_plot"].head(20):
        path = Path(str(value))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        assert path.exists(), f"absolute footprint plot is missing: {path}"
        assert path.stat().st_size > 1024


def test_context_safe_report_defaults_to_top_four_plotted_stacks():
    top = _require_report_table("top_hybrid_stacks_by_context.csv")
    manifest = _require_report_table("plot_manifest_layerwise_and_comparison.csv")

    assert pd.to_numeric(top["context_rank"], errors="coerce").max() == 4
    assert pd.to_numeric(manifest["context_rank"], errors="coerce").max() == 4
    assert len(manifest) == len(top), "plot manifest must include every selected top stack"

    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))
    assert 'TOP_K_PER_CONTEXT = int(globals().get("TOP_K_PER_CONTEXT", 4))' in source
    assert "Layerwise policy plot for this same stack" in source
    assert "Baseline model scale (checkpoint-derived)" in source
    assert "SHOW_ABSOLUTE_FOOTPRINT_PLOTS" in source
    assert "Refreshing context selection:" in source
    assert "build_context_safe_report_v4_from_checkpoints.py" in source
    assert "Refreshing checkpoint-derived metrics and absolute plots:" in source
    assert "Rendering final context-safe report:" in source
    assert "--skip-plots" in source


def test_top_hybrid_stacks_have_layerwise_policy_plots():
    manifest = _require_report_table("plot_manifest_layerwise_and_comparison.csv")
    top = _require_report_table("top_hybrid_stacks_by_context.csv")

    assert "layerwise_plot" in manifest.columns, "plot manifest lost the layerwise_plot column"
    plotted = manifest[manifest["layerwise_plot"].astype(str).str.len().gt(0)].copy()
    assert not plotted.empty, "no layerwise policy plots are listed in the manifest"

    merge_keys = [
        "objective",
        "dataset",
        "model",
        "scope",
        "ratio",
        "context_rank",
        "report_stack_id",
    ]
    required = top[[c for c in merge_keys if c in top.columns]].drop_duplicates()
    available = plotted[[c for c in merge_keys if c in plotted.columns]].drop_duplicates()
    merged = required.merge(available, on=[c for c in merge_keys if c in required.columns and c in available.columns], how="left", indicator=True)
    missing = merged[merged["_merge"].ne("both")]
    assert missing.empty, "top hybrid stacks are missing layerwise policy plots:\n" + missing.to_string(index=False)

    missing_files = []
    invalid_pngs = []
    for value in plotted["layerwise_plot"]:
        path = Path(str(value))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            missing_files.append(str(path))
            continue
        if path.stat().st_size < 1024:
            invalid_pngs.append(f"{path} is too small ({path.stat().st_size} bytes)")
            continue
        with path.open("rb") as handle:
            if handle.read(8) != b"\x89PNG\r\n\x1a\n":
                invalid_pngs.append(f"{path} is not a PNG")

    assert not missing_files, "layerwise plot files listed in the manifest are missing:\n" + "\n".join(missing_files[:20])
    assert not invalid_pngs, "layerwise plot files are invalid:\n" + "\n".join(invalid_pngs[:20])


def test_reporting_notebook_keeps_thesis_comparison_gallery():
    assert NOTEBOOK_PATH.exists(), f"Missing reporting notebook: {NOTEBOOK_PATH}"
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))

    assert "Thesis-ready hybrid-vs-singular comparison plots" in source
    assert "Hybrid vs same-context singular pruning methods" in source
    assert "comparison_plot" in source
    assert "Image(filename=str(plot_path))" in source
    assert "SHOW_COMPARISON_TABLES" in source
    assert 'MAX_INLINE_COMPARISON_PLOTS = globals().get("MAX_INLINE_COMPARISON_PLOTS", "all")' in source

    comparison_section = source.find("## 2. Hybrid vs singular comparison plots")
    layerwise_section = source.find("## 3. Layerwise policy plots")
    assert comparison_section >= 0, "comparison plot section is missing"
    assert layerwise_section >= 0, "layerwise plot section is missing"
    assert comparison_section < layerwise_section, "comparison plots must appear before layerwise policy plots"


def test_context_safe_ranking_uses_objective_champions():
    ranked = _require_report_table("all_ranked_hybrid_stacks_by_context.csv")
    assert "rank_selection_reason" in ranked.columns, "ranked table must explain why each stack was selected"

    required_cols = ["objective", "dataset", "model", "scope", "ratio", "context_rank", "rank_selection_reason"]
    for col in required_cols:
        assert col in ranked.columns, f"ranked table missing {col}"

    problems = []
    for key, group in ranked.groupby(["objective", "dataset", "model", "scope", "ratio"], dropna=False):
        objective, dataset, model, scope, ratio = key
        if len(group) < 2:
            continue
        top = group[pd.to_numeric(group["context_rank"], errors="coerce").isin([1, 2])]
        reasons = " + ".join(top["rank_selection_reason"].fillna("").astype(str))
        expected = []
        if objective == "flops_accuracy":
            expected = ["max_accuracy_retention", "max_flops_reduction"]
        elif objective == "time_accuracy":
            expected = ["max_accuracy_retention", "min_pruning_time"]
        elif objective == "time_flops":
            expected = ["min_pruning_time", "max_flops_reduction"]
        for reason in expected:
            if reason not in reasons:
                problems.append(
                    {
                        "objective": objective,
                        "dataset": dataset,
                        "model": model,
                        "scope": scope,
                        "ratio": ratio,
                        "missing_reason": reason,
                        "observed_reasons": reasons,
                    }
                )

    assert not problems, "objective ranking failed to include requested champions:\n" + pd.DataFrame(problems).to_string(index=False)
