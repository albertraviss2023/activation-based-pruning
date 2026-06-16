#!/usr/bin/env python
"""Build thesis-ready hybrid-vs-singular plots with absolute model scale.

The builder consumes the checkpoint-derived V4 comparison artifact. It does not
rerun pruning or infer metrics from percentage-only CSV files.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


CONTEXT_COLUMNS = [
    "objective",
    "objective_label",
    "dataset",
    "model",
    "scope",
    "ratio",
    "context_rank",
    "report_stack_id",
    "stack_id",
]


def safe_float(value, default=math.nan):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def slug(value) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return text or "item"


def display_stack_id(value) -> str:
    numeric = safe_float(value)
    if math.isfinite(numeric) and abs(numeric - round(numeric)) < 1e-9:
        return f"{int(round(numeric)):04d}"
    text = str(value).strip()
    return text[-4:].zfill(4) if text else "0000"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def stack_mask(frame: pd.DataFrame, stack: pd.Series) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column in ["objective", "dataset", "model", "scope", "report_stack_id"]:
        mask &= frame[column].astype(str).eq(str(stack.get(column)))
    mask &= np.isclose(
        pd.to_numeric(frame["ratio"], errors="coerce"),
        safe_float(stack.get("ratio")),
    )
    mask &= pd.to_numeric(frame["context_rank"], errors="coerce").eq(
        safe_float(stack.get("context_rank"))
    )
    return mask


def first_metric_value(frame: pd.DataFrame, metric: str, value_column: str) -> float:
    values = pd.to_numeric(
        frame.loc[frame["metric"].eq(metric), value_column],
        errors="coerce",
    ).dropna()
    return safe_float(values.iloc[0]) if not values.empty else math.nan


def build_stack_table(stack: pd.Series, comparison: pd.DataFrame) -> pd.DataFrame:
    """Create one method row containing all directly measured comparison metrics."""
    sub = comparison.loc[stack_mask(comparison, stack)].copy()
    if sub.empty:
        return pd.DataFrame()

    methods = (
        sub[["singular_method", "singular_method_display"]]
        .drop_duplicates()
        .sort_values("singular_method_display", kind="mergesort")
    )
    rows = []
    for _, method in methods.iterrows():
        method_rows = sub[
            sub["singular_method"].astype(str).eq(str(method["singular_method"]))
        ]
        first = method_rows.iloc[0]
        rows.append(
            {
                **{column: stack.get(column) for column in CONTEXT_COLUMNS},
                "singular_method": method["singular_method"],
                "singular_method_display": method["singular_method_display"],
                "singular_accuracy_delta_pp": first_metric_value(
                    method_rows, "accuracy_delta_pp", "singular_value"
                ),
                "singular_flops_reduction_pct": first_metric_value(
                    method_rows, "direct_flops_reduction_pct", "singular_value"
                ),
                "singular_remaining_flops_billions": safe_float(
                    first.get("singular_model_gops")
                ),
                "singular_remaining_params_m": safe_float(
                    first.get("singular_model_params_m")
                ),
                "singular_time_sec": first_metric_value(
                    method_rows, "time_sec", "singular_value"
                ),
                "hybrid_accuracy_delta_pp": first_metric_value(
                    method_rows, "accuracy_delta_pp", "hybrid_value"
                ),
                "hybrid_flops_reduction_pct": first_metric_value(
                    method_rows, "direct_flops_reduction_pct", "hybrid_value"
                ),
                "hybrid_remaining_flops_billions": safe_float(
                    first.get("hybrid_model_gops")
                ),
                "hybrid_remaining_params_m": safe_float(
                    first.get("hybrid_model_params_m")
                ),
                "hybrid_time_sec": first_metric_value(
                    method_rows, "time_sec", "hybrid_value"
                ),
                "baseline_flops_billions": safe_float(
                    first.get("hybrid_baseline_gops")
                ),
                "baseline_params_m": safe_float(first.get("hybrid_baseline_params_m")),
                "hybrid_metric_provenance": first.get(
                    "hybrid_absolute_metric_provenance"
                ),
                "singular_metric_provenance": first.get(
                    "singular_absolute_metric_provenance"
                ),
            }
        )
    return pd.DataFrame(rows)


def add_bar_labels(
    ax,
    bars,
    values,
    fmt,
    fontsize=6.5,
    *,
    label_type="edge",
    color="#111827",
    rotation=0,
):
    labels = [fmt.format(value) if math.isfinite(safe_float(value)) else "" for value in values]
    try:
        ax.bar_label(
            bars,
            labels=labels,
            padding=2 if label_type == "edge" else 0,
            fontsize=fontsize,
            label_type=label_type,
            color=color,
            rotation=rotation,
        )
    except Exception:
        pass


def plot_enhanced_comparison(
    stack: pd.Series,
    table: pd.DataFrame,
    output_dir: Path,
) -> Path | None:
    """Plot an aligned method-card comparison using directly measured metrics."""
    if table.empty:
        return None

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    singular = (
        table.sort_values("singular_method_display", kind="mergesort")
        .drop_duplicates("singular_method", keep="first")
        .reset_index(drop=True)
    )

    def first_value(column: str) -> float:
        values = pd.to_numeric(table[column], errors="coerce").dropna()
        return safe_float(values.iloc[0]) if not values.empty else math.nan

    hybrid = {
        "method": "Hybrid",
        "row_type": "hybrid",
        "accuracy": first_value("hybrid_accuracy_delta_pp"),
        "reduction": first_value("hybrid_flops_reduction_pct"),
        "flops": first_value("hybrid_remaining_flops_billions"),
        "params": first_value("hybrid_remaining_params_m"),
        "time": first_value("hybrid_time_sec"),
    }
    baseline_flops = first_value("baseline_flops_billions")
    baseline_params = first_value("baseline_params_m")
    baseline = {
        "method": "Baseline",
        "row_type": "baseline",
        "accuracy": 0.0,
        "reduction": 0.0,
        "flops": baseline_flops,
        "params": baseline_params,
        "time": math.nan,
    }
    rows = [
        baseline,
        hybrid,
        *[
            {
                "method": row["singular_method_display"],
                "row_type": "singular",
                "accuracy": safe_float(row["singular_accuracy_delta_pp"]),
                "reduction": safe_float(row["singular_flops_reduction_pct"]),
                "flops": safe_float(row["singular_remaining_flops_billions"]),
                "params": safe_float(row["singular_remaining_params_m"]),
                "time": safe_float(row["singular_time_sec"]),
            }
            for _, row in singular.iterrows()
        ],
    ]
    card = pd.DataFrame(rows)
    y = np.arange(len(card))
    row_colors = {
        "baseline": "#374151",
        "hybrid": "#E76F00",
        "singular": "#2F6FB5",
    }
    singular_better_color = "#2F6FB5"
    singular_worse_color = "#A9C7E8"

    figure_height = max(6.2, 0.52 * len(card) + 2.8)
    fig = plt.figure(figsize=(20.0, figure_height), facecolor="white")
    grid = fig.add_gridspec(
        1,
        6,
        width_ratios=[0.55, 1.18, 1.08, 1.18, 1.18, 1.20],
        left=0.045,
        right=0.985,
        bottom=0.12,
        top=0.80,
        wspace=0.18,
    )
    ax_method = fig.add_subplot(grid[0, 0])
    metric_axes = [fig.add_subplot(grid[0, index]) for index in range(1, 6)]

    # Subtle alternating row guides make it easy to trace a method across panels.
    for ax in [ax_method, *metric_axes]:
        for row_index in range(len(card)):
            if row_index % 2:
                ax.axhspan(
                    row_index - 0.5,
                    row_index + 0.5,
                    color="#F8FAFC",
                    zorder=0,
                )
        ax.set_ylim(len(card) - 0.5, -0.5)

    ax_method.set_xlim(0, 1)
    ax_method.axis("off")
    ax_method.set_title("Method", loc="left", fontsize=11, fontweight="bold", pad=14)
    for row_index, card_row in card.iterrows():
        ax_method.text(
            0.0,
            row_index,
            str(card_row["method"]),
            va="center",
            ha="left",
            fontsize=10.5,
            fontweight="bold"
            if card_row["row_type"] != "singular"
            else "semibold",
            color=row_colors[card_row["row_type"]],
        )

    specifications = [
        ("accuracy", "Accuracy delta", "(pp)", "{:.2f}", "higher"),
        ("reduction", "FLOPs reduction", "(%)", "{:.1f}%", "higher"),
        ("flops", "Remaining FLOPs", "(B)", "{:.3f}B", "lower"),
        ("params", "Remaining parameters", "(M)", "{:.2f}M", "lower"),
        ("time", "Pruning time", "(s)", "{:.1f}s", "lower"),
    ]

    def finite_extent(values: pd.Series) -> tuple[float, float]:
        finite = pd.to_numeric(values, errors="coerce")
        finite = finite[np.isfinite(finite)]
        if finite.empty:
            return 0.0, 1.0
        low = min(0.0, float(finite.min()))
        high = max(0.0, float(finite.max()))
        span = max(high - low, max(abs(low), abs(high), 1.0) * 0.08)
        return low - 0.04 * span, high + 0.22 * span

    for ax, (column, title, unit, value_format, better_direction) in zip(
        metric_axes, specifications
    ):
        values = pd.to_numeric(card[column], errors="coerce")
        drawable = values.fillna(0.0)
        hybrid_value = safe_float(
            card.loc[card["row_type"].eq("hybrid"), column].iloc[0]
        )
        panel_colors = []
        for _, card_row in card.iterrows():
            row_type = card_row["row_type"]
            value = safe_float(card_row[column])
            if row_type != "singular":
                panel_colors.append(row_colors[row_type])
                continue
            if not math.isfinite(value) or not math.isfinite(hybrid_value):
                panel_colors.append(singular_worse_color)
                continue
            worse_than_hybrid = (
                value < hybrid_value
                if better_direction == "higher"
                else value > hybrid_value
            )
            panel_colors.append(
                singular_worse_color
                if worse_than_hybrid
                else singular_better_color
            )
        bars = ax.barh(
            y,
            drawable,
            height=0.50,
            color=panel_colors,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        for row_index, (bar, value) in enumerate(zip(bars, values)):
            row_type = card.iloc[row_index]["row_type"]
            if row_type == "baseline" and column in {"accuracy", "reduction", "time"}:
                bar.set_alpha(0.0)
                ax.text(
                    0.02,
                    bar.get_y() + bar.get_height() / 2,
                    "reference" if column in {"accuracy", "reduction"} else "not applicable",
                    transform=ax.get_yaxis_transform(),
                    va="center",
                    ha="left",
                    fontsize=7.5,
                    color="#6B7280",
                )
                continue
            if not math.isfinite(safe_float(value)):
                bar.set_alpha(0.12)
                ax.text(
                    0.02,
                    bar.get_y() + bar.get_height() / 2,
                    "not available",
                    transform=ax.get_yaxis_transform(),
                    va="center",
                    ha="left",
                    fontsize=7.5,
                    color="#94A3B8",
                )
                continue
            x_value = safe_float(value)
            if x_value >= 0:
                x_text = x_value
                align = "left"
                offset = (5, 0)
            else:
                x_text = x_value
                align = "right"
                offset = (-5, 0)
            ax.annotate(
                value_format.format(x_value),
                xy=(x_text, bar.get_y() + bar.get_height() / 2),
                xytext=offset,
                textcoords="offset points",
                va="center",
                ha=align,
                fontsize=8.4,
                color="#111827",
            )

        ax.set_title(
            f"{title}\n{unit}",
            fontsize=10.8,
            fontweight="bold",
            pad=10,
            linespacing=1.3,
        )
        ax.set_yticks([])
        ax.grid(axis="x", color="#DCE3EA", linewidth=0.7, alpha=0.75, zorder=0)
        ax.axvline(0, color="#94A3B8", linewidth=0.9, zorder=2)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#94A3B8")
        ax.tick_params(axis="x", labelsize=8, colors="#475569")

        if column == "accuracy":
            ax.set_xlim(*finite_extent(values))
        elif column == "reduction":
            upper = max(100.0, safe_float(values.max(), 0.0) * 1.18)
            ax.set_xlim(0, upper)
        elif column == "flops":
            upper = max(
                safe_float(values.max(), 0.0),
                baseline_flops if math.isfinite(baseline_flops) else 0.0,
            )
            ax.set_xlim(0, max(upper * 1.18, 0.1))
            if math.isfinite(baseline_flops):
                ax.axvline(
                    baseline_flops,
                    color="#64748B",
                    linestyle="--",
                    linewidth=1.15,
                    zorder=2,
                )
                ax.text(
                    baseline_flops,
                    -0.72,
                    f"baseline {baseline_flops:.3f}B",
                    ha="right",
                    va="bottom",
                    fontsize=7.5,
                    color="#475569",
                    clip_on=False,
                )
        elif column == "params":
            upper = max(
                safe_float(values.max(), 0.0),
                baseline_params if math.isfinite(baseline_params) else 0.0,
            )
            ax.set_xlim(0, max(upper * 1.18, 1.0))
            if math.isfinite(baseline_params):
                ax.axvline(
                    baseline_params,
                    color="#64748B",
                    linestyle="--",
                    linewidth=1.15,
                    zorder=2,
                )
                ax.text(
                    baseline_params,
                    -0.72,
                    f"baseline {baseline_params:.2f}M",
                    ha="right",
                    va="bottom",
                    fontsize=7.5,
                    color="#475569",
                    clip_on=False,
                )
        elif column == "time":
            # Time remains on an ordinary linear scale. Exact labels retain the
            # magnitude of slow outliers without introducing a log transform.
            ax.set_xlim(*finite_extent(values))

    ratio = safe_float(stack.get("ratio"))
    fig.suptitle(
        "Hybrid vs same-context singular pruning methods",
        fontsize=18,
        fontweight="bold",
        color="#0F172A",
        y=0.965,
    )
    fig.text(
        0.5,
        0.905,
        f"Stack {display_stack_id(stack.get('report_stack_id'))} | rank {int(safe_float(stack.get('context_rank'), 0))} | "
        f"{stack.get('objective_label')} | {stack.get('dataset')} | {stack.get('model')} | "
        f"{str(stack.get('scope')).title()} scope | r={ratio:g}",
        ha="center",
        fontsize=11.5,
        color="#475569",
    )
    baseline_parts = []
    if math.isfinite(baseline_flops):
        baseline_parts.append(f"{baseline_flops:.3f}B FLOPs")
    if math.isfinite(baseline_params):
        baseline_parts.append(f"{baseline_params:.2f}M parameters")
    footer = (
        "Unpruned baseline: "
        + " and ".join(baseline_parts)
        + ". "
        if baseline_parts
        else ""
    )
    footer += (
        "Dark gray denotes the unpruned baseline and orange the hybrid stack. "
        "Strong blue singular bars match or outperform the hybrid for that metric; "
        "pale blue bars perform worse."
    )
    fig.text(
        0.5,
        0.035,
        footer,
        ha="center",
        fontsize=8.6,
        color="#475569",
    )

    path = output_dir / (
        f"enhanced_{slug(stack.get('objective'))}_{slug(stack.get('dataset'))}_"
        f"{slug(stack.get('model'))}_{slug(stack.get('scope'))}_r{ratio:g}_"
        f"rank{int(safe_float(stack.get('context_rank'), 0))}_{display_stack_id(stack.get('report_stack_id'))}.png"
    )
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--v4-report-dir",
        type=Path,
        default=Path(
            "report_artifacts/context_safe_hybrid_singular_report_v4_model_metrics"
        ),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path(
            "report_artifacts/context_safe_hybrid_singular_report_v5_enhanced_absolute_metrics"
        ),
    )
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-plots", type=int, default=None)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Render plots even when some singular checkpoint-derived absolute metrics are missing.",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    v4_dir = args.v4_report_dir
    if not v4_dir.is_absolute():
        v4_dir = (project_root / v4_dir).resolve()
    report_dir = args.report_dir
    if not report_dir.is_absolute():
        report_dir = (project_root / report_dir).resolve()
    table_dir = report_dir / "tables"
    plot_dir = report_dir / "plots" / "enhanced_comparisons"
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    for stale_plot in plot_dir.glob("enhanced_*.png"):
        stale_plot.unlink()

    comparison_path = (
        v4_dir / "tables" / "v4_hybrid_vs_singular_checkpoint_direct_long.csv"
    )
    comparison = read_csv(comparison_path)
    if comparison.empty:
        raise RuntimeError(
            "Checkpoint-derived V4 comparison table is missing or empty: "
            f"{comparison_path}"
        )

    stacks = (
        comparison[CONTEXT_COLUMNS]
        .drop_duplicates()
        .assign(
            context_rank=lambda frame: pd.to_numeric(
                frame["context_rank"], errors="coerce"
            ),
            ratio=lambda frame: pd.to_numeric(frame["ratio"], errors="coerce"),
        )
    )
    stacks = stacks[stacks["context_rank"].le(args.top_k)].sort_values(
        [
            "objective_label",
            "dataset",
            "model",
            "scope",
            "ratio",
            "context_rank",
            "report_stack_id",
        ],
        kind="mergesort",
    )
    if args.max_plots is not None:
        stacks = stacks.head(max(0, args.max_plots))

    tables = []
    manifest_rows = []
    for _, stack in stacks.iterrows():
        stack_table = build_stack_table(stack, comparison)
        if stack_table.empty:
            continue
        hybrid_complete = bool(
            stack_table[
                [
                    "hybrid_flops_reduction_pct",
                            "hybrid_remaining_flops_billions",
                    "hybrid_remaining_params_m",
                ]
            ]
            .notna()
            .all()
            .all()
        )
        singular_complete_mask = stack_table[
            [
                "singular_flops_reduction_pct",
                "singular_remaining_flops_billions",
                "singular_remaining_params_m",
            ]
        ].notna().all(axis=1)
        singular_complete = bool(singular_complete_mask.all())
        missing_singular_methods = " + ".join(
            stack_table.loc[~singular_complete_mask, "singular_method_display"]
            .dropna()
            .astype(str)
        )
        thesis_ready = hybrid_complete and singular_complete
        should_plot = hybrid_complete and (singular_complete or args.allow_partial)
        plot_path = (
            plot_enhanced_comparison(stack, stack_table, plot_dir)
            if should_plot
            else None
        )
        tables.append(stack_table)
        manifest_rows.append(
            {
                **{column: stack.get(column) for column in CONTEXT_COLUMNS},
                "stack_display_id": display_stack_id(stack.get("report_stack_id")),
                "plot": str(plot_path) if plot_path else "",
                "comparison_rows": len(stack_table),
                "all_hybrid_absolute_metrics_available": hybrid_complete,
                "all_singular_absolute_metrics_available": singular_complete,
                "thesis_ready": thesis_ready,
                "plot_status": (
                    "complete"
                    if thesis_ready
                    else "partial_allowed"
                    if should_plot
                    else "excluded_missing_absolute_metrics"
                ),
                "missing_singular_absolute_methods": missing_singular_methods,
            }
        )

    report_table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    manifest = pd.DataFrame(manifest_rows)
    report_table.to_csv(
        table_dir / "enhanced_hybrid_vs_singular_absolute_metrics.csv", index=False
    )
    manifest.to_csv(table_dir / "enhanced_plot_manifest.csv", index=False)
    manifest.loc[~manifest.get("thesis_ready", False).astype(bool)].to_csv(
        table_dir / "enhanced_plot_exclusion_audit.csv", index=False
    )
    baseline = read_csv(v4_dir / "tables" / "v4_baseline_model_scale.csv")
    baseline = baseline.rename(
        columns={
            "baseline_gops": "baseline_flops_billions",
            "baseline_gops_min": "baseline_flops_billions_min",
            "baseline_gops_max": "baseline_flops_billions_max",
            "baseline_gops_invariant": "baseline_flops_invariant",
        }
    ).drop(columns=["operation_count_convention"], errors="ignore")
    baseline.to_csv(table_dir / "baseline_model_scale.csv", index=False)

    print(f"Enhanced plots: {len(manifest)}")
    print(f"Enhanced comparison rows: {len(report_table)}")
    print(f"Report directory: {report_dir}")


if __name__ == "__main__":
    main()
