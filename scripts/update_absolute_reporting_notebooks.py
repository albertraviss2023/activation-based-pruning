"""Update the two context-safe reporting notebooks for absolute model scale."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"Notebook update marker not found: {old[:100]!r}")
    return text.replace(old, new, 1)


def update_v2() -> None:
    path = ROOT / "context_safe_hybrid_singular_reporting_v2_memory_safe.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))

    cell = notebook["cells"][3]
    source = "".join(cell["source"])
    source = replace_once(
        source,
        '    "QC summary": read_table("qc_summary.csv"),\n',
        '    "QC summary": read_table("qc_summary.csv"),\n'
        '    "Baseline model scale (checkpoint-derived)": read_table("baseline_model_scale.csv"),\n',
    )
    cell["source"] = source.splitlines(keepends=True)

    cell = notebook["cells"][5]
    source = "".join(cell["source"])
    source = replace_once(
        source,
        'SHOW_COMPARISON_TABLES = bool(globals().get("SHOW_COMPARISON_TABLES", False))\n',
        'SHOW_COMPARISON_TABLES = bool(globals().get("SHOW_COMPARISON_TABLES", False))\n'
        'SHOW_ABSOLUTE_FOOTPRINT_PLOTS = bool(globals().get("SHOW_ABSOLUTE_FOOTPRINT_PLOTS", True))\n',
    )
    source = replace_once(
        source,
        '        if "layerwise_plot" in row.index and str(row.get("layerwise_plot", "")):\n',
        '        if SHOW_ABSOLUTE_FOOTPRINT_PLOTS and "absolute_footprint_plot" in row.index and str(row.get("absolute_footprint_plot", "")):\n'
        '            absolute_plot_path = Path(row["absolute_footprint_plot"])\n'
        '            if not absolute_plot_path.is_absolute():\n'
        '                absolute_plot_path = PROJECT_ROOT / absolute_plot_path\n'
        '            display(Markdown("##### Absolute compute and parameter footprint for this same stack"))\n'
        '            if absolute_plot_path.exists():\n'
        '                display(Image(filename=str(absolute_plot_path)))\n'
        '                display(Markdown(f"Absolute footprint plot file: `{absolute_plot_path}`"))\n'
        '            else:\n'
        '                display(Markdown(f"Missing absolute footprint plot: `{absolute_plot_path}`"))\n'
        '        if "layerwise_plot" in row.index and str(row.get("layerwise_plot", "")):\n',
    )
    cell["source"] = source.splitlines(keepends=True)

    markdown = notebook["cells"][4]
    md_source = "".join(markdown["source"])
    if "absolute remaining compute" not in md_source:
        md_source += (
            "\n\nThe FLOPs panel also reports absolute remaining compute in GOp when a saved "
            "checkpoint is available. The dotted/annotated baseline scale uses one "
            "multiply-accumulate as one operation. Optional appendix plots show remaining "
            "GOp and parameter counts in millions for the same exact context.\n"
        )
    markdown["source"] = md_source.splitlines(keepends=True)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def update_v4() -> None:
    path = ROOT / "context_safe_hybrid_singular_reporting_v4_model_metrics.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))

    cell = notebook["cells"][1]
    source = "".join(cell["source"])
    source = replace_once(
        source,
        'PLOT_DIR = V4_REPORT_DIR / "plots" / "comparisons"\n',
        'PLOT_DIR = V4_REPORT_DIR / "plots" / "comparisons"\n'
        'ABSOLUTE_PLOT_DIR = V4_REPORT_DIR / "plots" / "absolute_footprints"\n',
    )
    cell["source"] = source.splitlines(keepends=True)

    cell = notebook["cells"][2]
    source = "".join(cell["source"])
    source = replace_once(
        source,
        'qc = read_table("v4_qc_summary.csv")\n',
        'qc = read_table("v4_qc_summary.csv")\n'
        'baseline_scale_df = read_table("v4_baseline_model_scale.csv")\n',
    )
    source = replace_once(
        source,
        'show_table("V4 QC summary", qc)\n',
        'show_table("V4 QC summary", qc)\n'
        'show_table("Baseline model scale", baseline_scale_df, ["dataset", "model", "direct_input_shape"])\n',
    )
    cell["source"] = source.splitlines(keepends=True)

    cell = notebook["cells"][3]
    source = "".join(cell["source"])
    source = replace_once(
        source,
        '        if not layerwise_direct_df.empty:\n',
        '        absolute_path = Path(str(row.get("absolute_footprint_plot", "")))\n'
        '        display(Markdown("#### Absolute compute and parameter footprint"))\n'
        '        if absolute_path.exists():\n'
        '            display(Image(filename=str(absolute_path)))\n'
        '        else:\n'
        '            display(Markdown(f"Missing absolute footprint plot: `{absolute_path}`"))\n'
        '        if not layerwise_direct_df.empty:\n',
    )
    cell["source"] = source.splitlines(keepends=True)

    markdown = notebook["cells"][0]
    md_source = "".join(markdown["source"])
    if "one multiply-accumulate" not in md_source:
        md_source += (
            "\n\nAbsolute compute is reported in GOp using the explicit convention that one "
            "multiply-accumulate is counted as one operation. Parameter counts are shown in millions.\n"
        )
    markdown["source"] = md_source.splitlines(keepends=True)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    update_v2()
    update_v4()
    print("Updated context-safe V2 and V4 reporting notebooks.")


if __name__ == "__main__":
    main()
