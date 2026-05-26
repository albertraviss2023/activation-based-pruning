"""Audit whether selected stacks realize the stated objective in final metrics.

This script reads the v4 checkpoint-derived report tables and writes audit CSVs.
It does not modify notebooks or report plots.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def norm_high(series: pd.Series) -> pd.Series:
    s = to_num(series)
    finite = s[np.isfinite(s)]
    if finite.empty:
        return pd.Series(0.5, index=s.index)
    lo, hi = finite.min(), finite.max()
    if abs(hi - lo) < 1e-12:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def norm_low(series: pd.Series) -> pd.Series:
    return 1.0 - norm_high(series)


def objective_terms(objective: str) -> set[str]:
    objective = str(objective)
    if objective == "flops_accuracy":
        return {"flops", "accuracy"}
    if objective == "time_accuracy":
        return {"time", "accuracy"}
    if objective == "time_flops":
        return {"time", "flops"}
    return {"flops", "time", "accuracy"}


def objective_score(frame: pd.DataFrame, objective: str) -> pd.Series:
    terms = objective_terms(objective)
    acc = norm_high(frame["accuracy_delta_pp"])
    flops = norm_high(frame["direct_flops_reduction_pct"])
    time = norm_low(frame["time_sec"])
    if terms == {"flops", "accuracy"}:
        return 0.60 * acc + 0.40 * flops
    if terms == {"time", "accuracy"}:
        return 0.60 * acc + 0.40 * time
    if terms == {"time", "flops"}:
        return 0.45 * acc + 0.30 * flops + 0.25 * time
    return 0.50 * acc + 0.25 * flops + 0.25 * time


def method_label(row: pd.Series) -> str:
    if str(row.get("record_type", "")) == "singular":
        return str(row.get("method_display", row.get("method", "singular")))
    return f"hybrid:{row.get('report_stack_id')}"


def id_text(value: object) -> str:
    try:
        f = float(value)
        if math.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
    except Exception:
        pass
    text = str(value)
    return "" if text.lower() in {"nan", "none"} else text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("report_artifacts/context_safe_hybrid_singular_report_v4_model_metrics"),
    )
    parser.add_argument("--accuracy-gate-pp", type=float, default=-7.0)
    parser.add_argument("--slow-ratio-threshold", type=float, default=5.0)
    args = parser.parse_args()

    report_dir = args.report_dir if args.report_dir.is_absolute() else PROJECT_ROOT / args.report_dir
    table_dir = report_dir / "tables"
    metrics = read_csv_safe(table_dir / "v4_checkpoint_direct_model_metrics.csv")
    layerwise = read_csv_safe(table_dir / "v4_hybrid_layerwise_policy_linked_to_direct_metrics.csv")
    if metrics.empty:
        raise RuntimeError(f"Missing metrics table: {table_dir / 'v4_checkpoint_direct_model_metrics.csv'}")

    work = metrics.copy()
    for col in ["ratio", "context_rank", "accuracy_delta_pp", "time_sec", "direct_flops_reduction_pct"]:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = to_num(work[col])
    work = work[work["metric_status"].astype(str).eq("ok")].copy()

    # Deduplicate singular rows repeated for multiple hybrid stack comparisons.
    singular_keys = ["objective", "dataset", "model", "scope", "ratio", "method"]
    singular = work[work["record_type"].astype(str).eq("singular")].copy()
    singular = singular.sort_values(["objective", "dataset", "model", "scope", "ratio", "method", "time_sec"])
    singular = singular.drop_duplicates([c for c in singular_keys if c in singular.columns], keep="first")
    hybrid = work[work["record_type"].astype(str).eq("hybrid")].copy()
    candidates = pd.concat([hybrid, singular], ignore_index=True, sort=False)

    chip_counts = pd.DataFrame()
    if not layerwise.empty and {"report_stack_id", "selected_method_display"}.issubset(layerwise.columns):
        chip_counts = (
            layerwise.assign(
                selected_method_display=layerwise["selected_method_display"].astype(str),
                report_stack_id=layerwise["report_stack_id"].map(id_text),
            )
            .groupby(["objective", "dataset", "model", "scope", "ratio", "context_rank", "report_stack_id"], dropna=False)["selected_method_display"]
            .apply(lambda s: int(s.str.upper().eq("CHIP").sum()))
            .reset_index(name="chip_layer_count")
        )
        chip_counts["ratio"] = to_num(chip_counts["ratio"])
        chip_counts["context_rank"] = to_num(chip_counts["context_rank"])

    rows = []
    context_keys = ["objective", "dataset", "model", "scope", "ratio"]
    for key, ctx in candidates.groupby(context_keys, dropna=False):
        objective, dataset, model, scope, ratio = key
        ctx = ctx.copy()
        ctx["objective_realized_score"] = objective_score(ctx, objective)
        ctx["accuracy_gate_passed"] = ctx["accuracy_delta_pp"] >= float(args.accuracy_gate_pp)
        feasible = ctx[ctx["accuracy_gate_passed"]].copy()
        if feasible.empty:
            feasible = ctx.copy()

        best_realized = feasible.sort_values(
            ["objective_realized_score", "accuracy_delta_pp", "direct_flops_reduction_pct", "time_sec"],
            ascending=[False, False, False, True],
            na_position="last",
        ).iloc[0]
        fastest_feasible = feasible.sort_values(["time_sec", "accuracy_delta_pp"], ascending=[True, False], na_position="last").iloc[0]
        highest_acc_feasible = feasible.sort_values(["accuracy_delta_pp", "time_sec"], ascending=[False, True], na_position="last").iloc[0]
        highest_flops_feasible = feasible.sort_values(["direct_flops_reduction_pct", "accuracy_delta_pp"], ascending=[False, False], na_position="last").iloc[0]

        singular_ctx = singular[
            (singular["objective"].astype(str) == str(objective))
            & (singular["dataset"].astype(str) == str(dataset))
            & (singular["model"].astype(str) == str(model))
            & (singular["scope"].astype(str) == str(scope))
            & np.isclose(singular["ratio"], float(ratio))
        ].copy()
        singular_median_time = float(singular_ctx["time_sec"].median()) if not singular_ctx.empty else math.nan
        singular_min_time = float(singular_ctx["time_sec"].min()) if not singular_ctx.empty else math.nan

        hybrids_ctx = hybrid[
            (hybrid["objective"].astype(str) == str(objective))
            & (hybrid["dataset"].astype(str) == str(dataset))
            & (hybrid["model"].astype(str) == str(model))
            & (hybrid["scope"].astype(str) == str(scope))
            & np.isclose(hybrid["ratio"], float(ratio))
        ].copy()
        for _, h in hybrids_ctx.iterrows():
            h_match = ctx[
                ctx.get("record_type", pd.Series(dtype=str)).astype(str).eq("hybrid")
                & ctx.get("report_stack_id", pd.Series(dtype=str)).map(id_text).eq(id_text(h.get("report_stack_id")))
            ]
            h_score = float(h_match["objective_realized_score"].iloc[0]) if not h_match.empty else math.nan
            faster_same_or_better_acc = feasible[
                (feasible["time_sec"] < h["time_sec"])
                & (feasible["accuracy_delta_pp"] >= h["accuracy_delta_pp"])
            ]
            faster_accuracy_gate = feasible[feasible["time_sec"] < h["time_sec"]]
            slow_vs_singular_median = (
                float(h["time_sec"]) / singular_median_time
                if np.isfinite(float(h["time_sec"])) and np.isfinite(singular_median_time) and singular_median_time > 0
                else math.nan
            )
            chip_layers = math.nan
            if not chip_counts.empty:
                cc = chip_counts[
                    (chip_counts["objective"].astype(str) == str(objective))
                    & (chip_counts["dataset"].astype(str) == str(dataset))
                    & (chip_counts["model"].astype(str) == str(model))
                    & (chip_counts["scope"].astype(str) == str(scope))
                    & np.isclose(chip_counts["ratio"], float(ratio))
                    & (chip_counts["report_stack_id"].map(id_text) == id_text(h.get("report_stack_id")))
                ]
                if not cc.empty:
                    chip_layers = float(cc["chip_layer_count"].iloc[0])

            flags = []
            if str(objective) == "time_accuracy":
                if not faster_same_or_better_acc.empty:
                    flags.append("faster_same_or_better_accuracy_candidate_exists")
                if np.isfinite(slow_vs_singular_median) and slow_vs_singular_median >= float(args.slow_ratio_threshold):
                    flags.append("very_slow_vs_singular_median")
                if np.isfinite(chip_layers) and chip_layers > 0 and np.isfinite(slow_vs_singular_median) and slow_vs_singular_median > 2:
                    flags.append("chip_used_in_slow_time_objective_stack")

            rows.append(
                {
                    "objective": objective,
                    "dataset": dataset,
                    "model": model,
                    "scope": scope,
                    "ratio": ratio,
                    "context_rank": h.get("context_rank"),
                    "report_stack_id": h.get("report_stack_id"),
                    "stack_id": h.get("stack_id"),
                    "selected_methods": h.get("selected_methods"),
                    "chip_layer_count": chip_layers,
                    "hybrid_accuracy_delta_pp": h.get("accuracy_delta_pp"),
                    "hybrid_time_sec": h.get("time_sec"),
                    "hybrid_direct_flops_reduction_pct": h.get("direct_flops_reduction_pct"),
                    "hybrid_objective_realized_score": h_score,
                    "best_realized_candidate": method_label(best_realized),
                    "best_realized_record_type": best_realized.get("record_type"),
                    "best_realized_accuracy_delta_pp": best_realized.get("accuracy_delta_pp"),
                    "best_realized_time_sec": best_realized.get("time_sec"),
                    "best_realized_direct_flops_reduction_pct": best_realized.get("direct_flops_reduction_pct"),
                    "fastest_feasible_candidate": method_label(fastest_feasible),
                    "fastest_feasible_record_type": fastest_feasible.get("record_type"),
                    "fastest_feasible_accuracy_delta_pp": fastest_feasible.get("accuracy_delta_pp"),
                    "fastest_feasible_time_sec": fastest_feasible.get("time_sec"),
                    "highest_accuracy_feasible_candidate": method_label(highest_acc_feasible),
                    "highest_accuracy_feasible_accuracy_delta_pp": highest_acc_feasible.get("accuracy_delta_pp"),
                    "highest_flops_feasible_candidate": method_label(highest_flops_feasible),
                    "highest_flops_feasible_direct_flops_reduction_pct": highest_flops_feasible.get("direct_flops_reduction_pct"),
                    "singular_method_count": int(len(singular_ctx)),
                    "singular_min_time_sec": singular_min_time,
                    "singular_median_time_sec": singular_median_time,
                    "hybrid_time_over_singular_median": slow_vs_singular_median,
                    "num_faster_same_or_better_accuracy_candidates": int(len(faster_same_or_better_acc)),
                    "num_faster_accuracy_gate_candidates": int(len(faster_accuracy_gate)),
                    "audit_flags": ";".join(flags),
                }
            )

    audit = pd.DataFrame(rows)
    out = table_dir / "v4_objective_realization_audit.csv"
    audit.to_csv(out, index=False)

    summary = (
        audit.assign(has_flag=audit["audit_flags"].fillna("").astype(str).ne(""))
        .groupby(["objective", "dataset", "model", "scope"], dropna=False)
        .agg(
            audited_hybrid_stacks=("report_stack_id", "count"),
            flagged_stacks=("has_flag", "sum"),
            mean_hybrid_time_over_singular_median=("hybrid_time_over_singular_median", "mean"),
            max_hybrid_time_over_singular_median=("hybrid_time_over_singular_median", "max"),
            mean_chip_layers=("chip_layer_count", "mean"),
        )
        .reset_index()
    )
    summary_out = table_dir / "v4_objective_realization_audit_summary.csv"
    summary.to_csv(summary_out, index=False)
    print(f"Saved {out}")
    print(f"Saved {summary_out}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
