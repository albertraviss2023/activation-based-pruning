"""Build a context-wise singular pruning report from saved singular checkpoints.

The report is grouped by dataset x model x scope x prune ratio. For each context
it tabulates singular-method accuracy delta, checkpoint-derived FLOPs reduction,
and checkpoint-derived parameter reduction, then writes three metric plots.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_context_safe_report_v4_from_checkpoints import (
    BaselineKey,
    checkpoint_metrics,
    method_display,
    read_csv_safe,
    safe_float,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_RATIOS = (0.3, 0.45, 0.55)
CONTEXT_KEYS = ["dataset", "model", "scope", "ratio"]


def slug(value: Any) -> str:
    import re

    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return text or "item"


def normalize_ratio(value: Any) -> float:
    try:
        return round(float(value), 8)
    except Exception:
        return math.nan


def load_singular_index(registry_dir: Path, allowed_ratios: tuple[float, ...]) -> pd.DataFrame:
    cache = read_csv_safe(registry_dir / "singular_cache_index.csv")
    if cache.empty:
        raise RuntimeError(f"Missing singular cache index: {registry_dir / 'singular_cache_index.csv'}")
    cache = cache.copy()
    cache["ratio"] = pd.to_numeric(cache["ratio"], errors="coerce")
    cache = cache[cache["ratio"].apply(lambda r: any(np.isclose(r, a) for a in allowed_ratios))].copy()
    cache["method_display"] = cache["method"].map(method_display)
    cache["record_type"] = "singular"
    cache["checkpoint_path_original"] = cache.get("checkpoint_path", "")
    cache["checkpoint_path"] = cache.get("checkpoint_path", "")
    cache["checkpoint_path_resolved"] = ""
    cache["report_stack_id"] = ""
    cache["context_rank"] = np.nan
    cache["stack_id"] = cache["method"].astype(str)
    cache["selected_methods"] = cache["method"].astype(str)
    return cache


def choose_latest_per_context_method(cache: pd.DataFrame) -> pd.DataFrame:
    work = cache.copy()
    if "run_modified_utc" in work.columns:
        work["_sort_time"] = pd.to_datetime(work["run_modified_utc"], errors="coerce")
    else:
        work["_sort_time"] = pd.NaT
    if "timestamp" in work.columns:
        work["_sort_stamp"] = work["timestamp"].astype(str)
    else:
        work["_sort_stamp"] = ""
    work = work.sort_values(
        ["dataset", "model", "scope", "ratio", "method", "_sort_time", "_sort_stamp"],
        ascending=[True, True, True, True, True, False, False],
        na_position="last",
    )
    return work.drop_duplicates(["dataset", "model", "scope", "ratio", "method"], keep="first").drop(
        columns=["_sort_time", "_sort_stamp"],
        errors="ignore",
    )


def profile_singular_checkpoints(index: pd.DataFrame, project_root: Path, device: str) -> pd.DataFrame:
    baseline_cache: dict[BaselineKey, tuple[float, float]] = {}
    rows = []
    for _, row in index.iterrows():
        rows.append(checkpoint_metrics(row, project_root, device, baseline_cache))
    metrics = pd.DataFrame(rows)

    # Preserve benchmark accuracy/time fields from the singular index, while using
    # checkpoint-derived structural FLOPs/params from the loaded model.
    keep_cols = [
        "dataset",
        "model",
        "scope",
        "ratio",
        "method",
        "accuracy_pct",
        "baseline_accuracy_pct",
        "accuracy_delta_pp",
        "time_sec",
        "run_id",
        "run_dir",
        "source_table",
        "cache_key",
        "checkpoint_path",
    ]
    extra = index[[c for c in keep_cols if c in index.columns]].copy()
    merged = metrics.merge(
        extra,
        on=["dataset", "model", "scope", "ratio", "method"],
        how="left",
        suffixes=("", "_artifact"),
    )
    if "accuracy_delta_pp_artifact" in merged.columns:
        merged["accuracy_delta_pp"] = merged["accuracy_delta_pp_artifact"]
    if "accuracy_pct_artifact" in merged.columns:
        merged["accuracy_pct"] = merged["accuracy_pct_artifact"]
    if "baseline_accuracy_pct_artifact" in merged.columns:
        merged["baseline_accuracy_pct"] = merged["baseline_accuracy_pct_artifact"]
    if "time_sec_artifact" in merged.columns:
        merged["time_sec"] = merged["time_sec_artifact"]
    return merged


def write_context_plots(metrics: pd.DataFrame, plot_dir: Path) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_rows = []
    ok = metrics[metrics["metric_status"].astype(str).eq("ok")].copy()
    for key, group in ok.groupby(CONTEXT_KEYS, dropna=False):
        dataset, model, scope, ratio = key
        group = group.copy()
        group["method_display"] = group["method"].map(method_display)
        context_label = f"{dataset} | {model} | {scope} | r={float(ratio):g}"
        context_slug = f"{slug(dataset)}_{slug(model)}_{slug(scope)}_r{float(ratio):g}"
        panels = [
            ("accuracy_delta_pp", "Accuracy delta vs baseline (pp)", "#10B981", False, "accuracy_delta"),
            ("direct_flops_reduction_pct", "Direct structural FLOPs reduction (%)", "#2563EB", False, "flops_reduction"),
            ("direct_params_reduction_pct", "Direct parameter reduction (%)", "#7C3AED", False, "params_reduction"),
        ]

        combined_path = plot_dir / f"singular_context_metrics_{context_slug}.png"
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.0))
        metric_paths = {}
        for ax, (metric, ylabel, color, lower_is_better, short_name) in zip(axes, panels):
            sub = group.dropna(subset=[metric]).sort_values(metric, ascending=lower_is_better)
            bars = ax.bar(sub["method_display"], sub[metric], color=color, alpha=0.82)
            try:
                ax.bar_label(bars, labels=[f"{v:.2f}" for v in sub[metric]], padding=2, fontsize=7)
            except Exception:
                pass
            if metric == "accuracy_delta_pp":
                ax.axhline(0, color="#64748B", linewidth=0.8)
            ax.set_title(short_name.replace("_", " "))
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", rotation=55)
            ax.grid(axis="y", alpha=0.25)

            single_path = plot_dir / f"singular_{short_name}_{context_slug}.png"
            sfig, sax = plt.subplots(figsize=(8.5, 4.8))
            sbars = sax.bar(sub["method_display"], sub[metric], color=color, alpha=0.82)
            try:
                sax.bar_label(sbars, labels=[f"{v:.2f}" for v in sub[metric]], padding=2, fontsize=7)
            except Exception:
                pass
            if metric == "accuracy_delta_pp":
                sax.axhline(0, color="#64748B", linewidth=0.8)
            sax.set_title(f"{short_name.replace('_', ' ')} | {context_label}")
            sax.set_ylabel(ylabel)
            sax.tick_params(axis="x", rotation=55)
            sax.grid(axis="y", alpha=0.25)
            sfig.tight_layout()
            sfig.savefig(single_path, dpi=180, bbox_inches="tight")
            plt.close(sfig)
            metric_paths[short_name] = str(single_path)

        fig.suptitle(f"Singular methods by context | {context_label}", fontsize=12)
        fig.tight_layout()
        fig.savefig(combined_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        plot_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "scope": scope,
                "ratio": ratio,
                "method_count": int(len(group)),
                "combined_plot": str(combined_path),
                "accuracy_delta_plot": metric_paths.get("accuracy_delta", ""),
                "flops_reduction_plot": metric_paths.get("flops_reduction", ""),
                "params_reduction_plot": metric_paths.get("params_reduction", ""),
            }
        )
    return pd.DataFrame(plot_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--registry-dir", type=Path, default=Path("reports/experiment_registry"))
    parser.add_argument("--report-dir", type=Path, default=Path("report_artifacts/singular_method_context_report"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    registry_dir = args.registry_dir if args.registry_dir.is_absolute() else project_root / args.registry_dir
    report_dir = args.report_dir if args.report_dir.is_absolute() else project_root / args.report_dir
    table_dir = report_dir / "tables"
    plot_dir = report_dir / "plots"
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    singular_index_raw = load_singular_index(registry_dir, ALLOWED_RATIOS)
    singular_index = choose_latest_per_context_method(singular_index_raw)
    singular_index.to_csv(table_dir / "singular_method_checkpoint_index.csv", index=False)

    metrics = profile_singular_checkpoints(singular_index, project_root, args.device)
    metrics.to_csv(table_dir / "singular_method_context_metrics.csv", index=False)

    plot_manifest = write_context_plots(metrics, plot_dir)
    plot_manifest.to_csv(table_dir / "plot_manifest_singular_method_context_metrics.csv", index=False)

    coverage = (
        metrics.groupby(CONTEXT_KEYS, dropna=False)
        .agg(
            methods=("method", "count"),
            profiled_methods=("metric_status", lambda s: int((s.astype(str) == "ok").sum())),
            missing_or_failed_methods=("metric_status", lambda s: int((s.astype(str) != "ok").sum())),
        )
        .reset_index()
    )
    coverage.to_csv(table_dir / "singular_method_context_coverage.csv", index=False)

    failures = metrics[metrics["metric_status"].astype(str).ne("ok")].copy()
    failures.to_csv(table_dir / "singular_method_missing_or_failed_checkpoints.csv", index=False)

    manifest = {
        "report_dir": str(report_dir),
        "tables_dir": str(table_dir),
        "plots_dir": str(plot_dir),
        "counts": {
            "raw_singular_cache_rows": int(len(singular_index_raw)),
            "unique_context_method_rows": int(len(singular_index)),
            "metrics_rows": int(len(metrics)),
            "metrics_ok_rows": int(metrics["metric_status"].astype(str).eq("ok").sum()),
            "context_plot_rows": int(len(plot_manifest)),
            "missing_or_failed_rows": int(len(failures)),
        },
        "notes": [
            "FLOPs and params are recomputed from saved singular model checkpoints.",
            "Accuracy delta and pruning time come from the benchmark artifact index.",
            "Rows without resolvable checkpoint paths are listed in singular_method_missing_or_failed_checkpoints.csv.",
        ],
    }
    (report_dir / "singular_method_context_report_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
