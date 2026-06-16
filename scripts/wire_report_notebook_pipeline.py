"""Wire V2 reporting to refresh checkpoint metrics before rendering plots."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "context_safe_hybrid_singular_reporting_v2_memory_safe.ipynb"


def main() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = notebook["cells"][1]
    source = """# ============================================================
# 0. Build the complete context-safe report pipeline
# ============================================================
from pathlib import Path
import json
import subprocess
import sys

import pandas as pd
import numpy as np
from IPython.display import display, Markdown, Image

PROJECT_ROOT = Path.cwd()
REPORT_DIR = PROJECT_ROOT / "report_artifacts" / "context_safe_hybrid_singular_report"
V4_REPORT_DIR = PROJECT_ROOT / "report_artifacts" / "context_safe_hybrid_singular_report_v4_model_metrics"
TABLE_DIR = REPORT_DIR / "tables"
FIG_DIR = REPORT_DIR / "plots"

RUN_REGISTRY_REFRESH = bool(globals().get("RUN_REGISTRY_REFRESH", True))
RUN_CHECKPOINT_METRIC_REFRESH = bool(globals().get("RUN_CHECKPOINT_METRIC_REFRESH", True))
FORCE_CHECKPOINT_REPROFILE = bool(globals().get("FORCE_CHECKPOINT_REPROFILE", False))
TOP_K_PER_CONTEXT = int(globals().get("TOP_K_PER_CONTEXT", 4))
ACCURACY_GATE_PP = float(globals().get("ACCURACY_GATE_PP", 7.0))
MAX_PLOTS = globals().get("MAX_PLOTS", None)
MAX_PLOTS_ARG = [] if MAX_PLOTS in (None, "", "all", "ALL") else ["--max-plots", str(int(MAX_PLOTS))]

base_cmd = [
    sys.executable,
    str(PROJECT_ROOT / "scripts" / "build_context_safe_report.py"),
    "--project-root", str(PROJECT_ROOT),
    "--outputs-root", str(PROJECT_ROOT / "outputs" / "lfpc_hybrid"),
    "--registry-dir", str(PROJECT_ROOT / "reports" / "experiment_registry"),
    "--report-dir", str(REPORT_DIR),
    "--v4-report-dir", str(V4_REPORT_DIR),
    "--top-k", str(TOP_K_PER_CONTEXT),
    "--accuracy-gate-pp", str(ACCURACY_GATE_PP),
]

# Bootstrap the current top-stack selection before checkpoint profiling. This
# pass writes tables only, so plots are not rendered twice.
top_table = REPORT_DIR / "tables" / "top_hybrid_stacks_by_context.csv"
if RUN_REGISTRY_REFRESH or not top_table.exists():
    bootstrap_cmd = [*base_cmd, "--skip-plots"]
    if RUN_REGISTRY_REFRESH:
        bootstrap_cmd.append("--refresh-registry")
    print("Refreshing context selection:", " ".join(bootstrap_cmd))
    subprocess.run(bootstrap_cmd, check=True)

# Recompute report metrics from saved checkpoints. Existing checkpoint
# measurements are reused unless FORCE_CHECKPOINT_REPROFILE is enabled.
if RUN_CHECKPOINT_METRIC_REFRESH:
    v4_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "build_context_safe_report_v4_from_checkpoints.py"),
        "--project-root", str(PROJECT_ROOT),
        "--base-report-dir", str(REPORT_DIR),
        "--registry-dir", str(PROJECT_ROOT / "reports" / "experiment_registry"),
        "--report-dir", str(V4_REPORT_DIR),
        "--top-k", str(TOP_K_PER_CONTEXT),
    ]
    if FORCE_CHECKPOINT_REPROFILE:
        v4_cmd.append("--force-reprofile")
    print("Refreshing checkpoint-derived metrics and absolute plots:", " ".join(v4_cmd))
    subprocess.run(v4_cmd, check=True)

# Final pass produces all percentage, absolute-footprint, layerwise, Pareto,
# and summary plots from the freshly aligned tables.
final_cmd = [*base_cmd, *MAX_PLOTS_ARG]
print("Rendering final context-safe report:", " ".join(final_cmd))
subprocess.run(final_cmd, check=True)

manifest_path = REPORT_DIR / "report_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
print(json.dumps(manifest, indent=2))
"""
    cell["source"] = source.splitlines(keepends=True)

    markdown = notebook["cells"][0]
    md_source = "".join(markdown["source"])
    note = (
        "\n\nOn every normal run, the notebook first refreshes V4 checkpoint-derived "
        "absolute metrics and figures, then rebuilds the V2 context-safe report. Existing "
        "checkpoint measurements are reused by default; set "
        "`FORCE_CHECKPOINT_REPROFILE=True` only when the saved model files changed.\n"
    )
    if note not in md_source:
        markdown["source"] = (md_source + note).splitlines(keepends=True)

    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wired report pipeline in {NOTEBOOK.name}")


if __name__ == "__main__":
    main()
