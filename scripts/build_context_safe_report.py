"""Build context-safe LFPC hybrid/singular reporting artifacts.

This script is intentionally analysis-facing: it refreshes the experiment
registry, ranks hybrid stacks only within matching contexts, compares against
singular benchmarks only when dataset/model/scope/ratio match, and exports
tables plus figures that the reporting notebook can display inline.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COARSE_CONTEXT_KEYS = ["objective", "dataset", "model", "scope", "ratio"]
EXACT_CONTEXT_KEYS = COARSE_CONTEXT_KEYS + [
    "variance_threshold",
    "spearman_threshold",
    "jaccard_threshold",
]
DEFAULT_ALLOWED_PRUNE_RATIOS = (0.30, 0.45, 0.55)

METHOD_ALIASES = {
    "l1_norm": "L1",
    "custom_l2": "L2",
    "mean_abs_act": "MeanAct",
    "apoz": "APoZ",
    "custom_entropy": "Entropy",
    "custom_class_entropy": "ClassEntropy",
    "custom_hrank": "HRank",
    "custom_spectral_energy": "Spectral",
    "custom_reprune": "REPrune",
    "custom_tis": "TIS",
    "custom_nisp": "NISP",
    "chip": "CHIP",
    "custom_autodfp": "AutoDFP",
    "custom_gfi_ap": "GFI-AP",
    "custom_thinet": "ThiNet",
    "custom_dcp": "DCP",
    "custom_senpips": "SeNPIPS",
    "custom_gfs": "GFS",
}


def read_csv_safe(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def parse_allowed_ratios(value: Any) -> set[float]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        raw_items = value
    else:
        text = str(value).strip()
        if not text or text.lower() in {"all", "none", "*"}:
            return set()
        raw_items = re.split(r"[,;\s]+", text)
    ratios = set()
    for item in raw_items:
        ratio = safe_float(item)
        if math.isfinite(ratio):
            ratios.add(round(float(ratio), 6))
    return ratios


def ratio_allowed(value: Any, allowed_ratios: set[float]) -> bool:
    if not allowed_ratios:
        return True
    ratio = safe_float(value)
    return math.isfinite(ratio) and round(float(ratio), 6) in allowed_ratios


def safe_slug(value: Any, max_len: int = 140) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return (text or "item")[:max_len]


def resolve_path(project_root: Path, value: Any) -> Path | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    path = Path(str(value))
    return path if path.is_absolute() else project_root / path


def method_display(method: Any) -> str:
    raw = norm_text(method)
    return METHOD_ALIASES.get(raw, raw.replace("custom_", "").replace("_", " ").title())


def parse_literal(value: Any, default: Any = None) -> Any:
    if default is None:
        default = {}
    if isinstance(value, (dict, list, tuple)):
        return value
    if not norm_text(value) or norm_text(value).lower() == "nan":
        return default
    for loader in (ast.literal_eval, json.loads):
        try:
            return loader(str(value))
        except Exception:
            pass
    return default


def layer_sort_key(layer: Any) -> tuple[int, list[int], str]:
    text = str(layer)
    nums = [int(x) for x in re.findall(r"\d+", text)]
    prefix = 0 if text == "conv1" or text.startswith("features") else 1
    return (prefix, nums, text)


def assign_regions(layers: list[str]) -> dict[str, str]:
    ordered = sorted([str(x) for x in layers], key=layer_sort_key)
    total = len(ordered)
    regions: dict[str, str] = {}
    for idx, layer in enumerate(ordered):
        frac = (idx + 0.5) / max(total, 1)
        regions[layer] = "Early" if frac <= 1 / 3 else "Middle" if frac <= 2 / 3 else "Late"
    return regions


def norm_high(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    lo, hi = values.min(), values.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return pd.Series(0.5, index=values.index)
    return (values - lo) / (hi - lo)


def norm_low(series: pd.Series) -> pd.Series:
    return 1.0 - norm_high(series)


def objective_terms(objective: Any) -> set[str]:
    text = norm_text(objective)
    if text == "flops_accuracy":
        return {"flops", "accuracy"}
    if text == "time_accuracy":
        return {"time", "accuracy"}
    if text == "time_flops":
        return {"time", "flops"}
    return {"flops", "time", "accuracy"}


def stack_display_id(row: pd.Series) -> str:
    methods = parse_literal(row.get("selected_methods"), default=[])
    if isinstance(methods, dict):
        methods = list(methods.values())
    if not isinstance(methods, (list, tuple)) or not methods:
        methods = [row.get("method_or_stack", row.get("stack_id", "stack"))]
    cleaned = [method_display(m).replace("-", "").replace(" ", "") for m in methods if norm_text(m)]
    dominant = list(pd.Series(cleaned).value_counts().index[:3]) if cleaned else ["STACK"]
    prefix = "".join(x.upper()[:6] for x in dominant)[:18] or "STACK"
    key = str(row.get("stack_id", row.get("method_or_stack", ""))) + "|" + "+".join(cleaned)
    digest = hashlib.blake2s(key.encode("utf-8"), digest_size=4).hexdigest()
    return str(1000 + (int(digest, 16) % 9000))


def assign_unique_four_digit_stack_ids(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    used: set[str] = set()
    ids: list[str] = []
    for _, row in out.iterrows():
        key = "|".join(str(row.get(c, "")) for c in EXACT_CONTEXT_KEYS + ["stack_id", "method_or_stack", "selected_methods"])
        salt = 0
        while True:
            raw = hashlib.blake2s(f"{key}|{salt}".encode("utf-8"), digest_size=4).hexdigest()
            digest = str(1000 + (int(raw, 16) % 9000))
            if digest not in used:
                used.add(digest)
                ids.append(digest)
                break
            salt += 1
    out["report_stack_id"] = ids
    return out


def score_group(group: pd.DataFrame, accuracy_gate_pp: float) -> pd.DataFrame:
    out = group.copy()
    for col in ["accuracy_delta_pp", "accuracy_pct", "flops_reduction_pct", "time_sec"]:
        out[col] = pd.to_numeric(out.get(col, np.nan), errors="coerce")
    acc_score = norm_high(out["accuracy_delta_pp"].where(out["accuracy_delta_pp"].notna(), out["accuracy_pct"]))
    flops_score = norm_high(out["flops_reduction_pct"])
    time_score = norm_low(out["time_sec"])
    terms = objective_terms(out["objective"].iloc[0] if "objective" in out.columns and len(out) else "all_three")
    
    # Objective-aware continuous score used for tie-breaking and all-three ranking.
    # Pairwise top-k selection below additionally forces the requested objective
    # champions to appear in the top rows for each exact comparable context.
    if terms == {"flops", "accuracy"}:
        score = 0.50 * acc_score + 0.50 * flops_score
    elif terms == {"time", "accuracy"}:
        score = 0.50 * acc_score + 0.50 * time_score
    elif terms == {"time", "flops"}:
        score = 0.50 * flops_score + 0.50 * time_score
    else:
        # all_three
        score = (acc_score + flops_score + time_score) / 3.0
        
    out["analysis_score"] = score
    out["accuracy_gate_passed"] = out["accuracy_delta_pp"] >= -float(accuracy_gate_pp)
    return out


def objective_champion_order(scored: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Order stacks so the top rows answer the objective-specific question.

    Pairwise objectives are not treated as generic weighted averages. The top
    two rows deliberately expose the best observed metric champions inside the
    same objective x dataset x model x scope x prune-ratio context:
    - FLOPs + Accuracy: max accuracy retention and max FLOPs reduction.
    - Time + Accuracy: max accuracy retention and minimum pruning time.
    - Time + FLOPs: minimum pruning time and max FLOPs reduction.

    If a champion is the same stack, the remaining slots are filled by the
    balanced score. All selection prefers stacks passing the accuracy gate; if
    none pass, it falls back to all stacks so incomplete contexts still report.
    """
    if scored.empty:
        return scored

    work = scored.copy()
    for col in ["accuracy_delta_pp", "accuracy_pct", "flops_reduction_pct", "time_sec", "analysis_score"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    if "accuracy_gate_passed" not in work.columns:
        work["accuracy_gate_passed"] = False

    eligible = work[work["accuracy_gate_passed"].astype(bool)].copy()
    if eligible.empty:
        eligible = work.copy()

    terms = objective_terms(work["objective"].iloc[0] if "objective" in work.columns and len(work) else "all_three")
    selected: list[Any] = []
    reasons: dict[Any, str] = {}

    def add_best(sort_cols: list[str], ascending: list[bool], reason: str) -> None:
        candidates = eligible.dropna(subset=[sort_cols[0]]).copy() if sort_cols else eligible.copy()
        if candidates.empty:
            candidates = eligible.copy()
        if candidates.empty:
            return
        ordered = candidates.sort_values(sort_cols, ascending=ascending, na_position="last", kind="mergesort")
        idx = ordered.index[0]
        if idx not in selected:
            selected.append(idx)
            reasons[idx] = reason
        elif reason not in str(reasons.get(idx, "")).split("+"):
            reasons[idx] = f"{reasons[idx]}+{reason}"

    if terms == {"flops", "accuracy"}:
        add_best(["accuracy_delta_pp", "flops_reduction_pct", "time_sec"], [False, False, True], "max_accuracy_retention")
        add_best(["flops_reduction_pct", "accuracy_delta_pp", "time_sec"], [False, False, True], "max_flops_reduction")
    elif terms == {"time", "accuracy"}:
        add_best(["accuracy_delta_pp", "time_sec", "flops_reduction_pct"], [False, True, False], "max_accuracy_retention")
        add_best(["time_sec", "accuracy_delta_pp", "flops_reduction_pct"], [True, False, False], "min_pruning_time")
    elif terms == {"time", "flops"}:
        add_best(["time_sec", "flops_reduction_pct", "accuracy_delta_pp"], [True, False, False], "min_pruning_time")
        add_best(["flops_reduction_pct", "time_sec", "accuracy_delta_pp"], [False, True, False], "max_flops_reduction")
    else:
        # Balanced all-three objective: accuracy high, FLOPs high, time low.
        pass

    fill = work.sort_values(
        ["accuracy_gate_passed", "analysis_score", "accuracy_delta_pp", "flops_reduction_pct", "time_sec"],
        ascending=[False, False, False, False, True],
        na_position="last",
        kind="mergesort",
    )
    for idx in fill.index:
        if idx not in selected:
            selected.append(idx)
            reasons[idx] = "balanced_objective_score"

    ordered = work.loc[selected].copy()
    ordered["rank_selection_reason"] = [reasons.get(idx, "balanced_objective_score") for idx in ordered.index]
    return ordered


def context_key(row: pd.Series | dict[str, Any]) -> str:
    return "|".join(str(row.get(k, "")) for k in COARSE_CONTEXT_KEYS)


def registry_style_context_key(row: pd.Series | dict[str, Any]) -> str:
    """Match build_experiment_registry.py's exact context key format."""
    parts = [
        row.get("dataset", ""),
        row.get("model", ""),
        row.get("objective", ""),
        row.get("scope", ""),
        f"r{safe_float(row.get('ratio')):g}" if math.isfinite(safe_float(row.get("ratio"))) else "rNA",
        f"v{safe_float(row.get('variance_threshold')):g}" if math.isfinite(safe_float(row.get("variance_threshold"))) else "vNA",
        f"s{safe_float(row.get('spearman_threshold')):g}" if math.isfinite(safe_float(row.get("spearman_threshold"))) else "sNA",
        f"j{safe_float(row.get('jaccard_threshold')):g}" if math.isfinite(safe_float(row.get("jaccard_threshold"))) else "jNA",
    ]
    return "__".join(safe_slug(p) for p in parts)


def filter_to_allowed_ratios(frame: pd.DataFrame, allowed_ratios: set[float], label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop stale/legacy contexts whose prune ratio is outside the experiment grid."""
    if frame.empty or "ratio" not in frame.columns or not allowed_ratios:
        return frame, pd.DataFrame()
    ratios = pd.to_numeric(frame["ratio"], errors="coerce").round(6)
    keep = ratios.isin(sorted(allowed_ratios))
    excluded = frame.loc[~keep].copy()
    audit_rows = [
        {
            "artifact": label,
            "status": "applied",
            "allowed_ratios": ", ".join(f"{r:g}" for r in sorted(allowed_ratios)),
            "rows_before": int(len(frame)),
            "rows_after": int(keep.sum()),
            "rows_excluded": int((~keep).sum()),
        }
    ]
    if not excluded.empty:
        cols = [
            c
            for c in [
                "record_type",
                "run_id",
                "timestamp",
                "dataset",
                "model",
                "objective",
                "scope",
                "ratio",
                "variance_threshold",
                "spearman_threshold",
                "jaccard_threshold",
                "method",
                "stack_id",
                "source_table",
                "source_row",
            ]
            if c in excluded.columns
        ]
        details = excluded[cols].copy()
        details.insert(0, "status", "excluded_disallowed_prune_ratio")
        details.insert(0, "artifact", label)
        details["allowed_ratios"] = ", ".join(f"{r:g}" for r in sorted(allowed_ratios))
        audit_rows.extend(details.to_dict(orient="records"))
    return frame.loc[keep].copy(), pd.DataFrame(audit_rows)


def filter_hybrids_to_latest_family_runs(contexts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep hybrid rows only from the latest run per objective/dataset/model/scope/ratio.

    We use the full coarse context (including scope and ratio) to ensure that
    running a new experiment for one specific ratio doesn't accidentally
    discard valid results for other ratios from an earlier run in the same family.
    """
    if contexts.empty:
        return contexts, pd.DataFrame()
    required = {"record_type", "objective", "dataset", "model", "scope", "ratio", "run_id"}
    if not required.issubset(contexts.columns):
        return contexts, pd.DataFrame([{"status": "not_applied", "issue": f"missing columns: {sorted(required - set(contexts.columns))}"}])

    out = contexts.copy()
    hybrid_mask = out["record_type"].astype(str).eq("hybrid")
    hybrids = out.loc[hybrid_mask].copy()
    if hybrids.empty:
        return out, pd.DataFrame([{"status": "not_applied", "issue": "no hybrid rows"}])

    if "timestamp" not in hybrids.columns:
        hybrids["timestamp"] = ""
    if "run_modified_utc" not in hybrids.columns:
        hybrids["run_modified_utc"] = ""

    # Scope the "latest run" logic to the actual reporting context to avoid over-aggressive filtering.
    family_cols = ["objective", "dataset", "model", "scope", "ratio"]
    latest_run_ids: set[str] = set()
    summary_rows: list[dict[str, Any]] = []
    for keys, group in hybrids.groupby(family_cols, dropna=False):
        run_summary = (
            group.groupby("run_id", dropna=False)
            .agg(
                timestamp=("timestamp", "max"),
                run_modified_utc=("run_modified_utc", "max"),
                hybrid_rows=("run_id", "size"),
            )
            .reset_index()
            .sort_values(["timestamp", "run_modified_utc", "run_id"], ascending=[False, False, False])
        )
        if run_summary.empty:
            continue
        latest_run = str(run_summary.iloc[0]["run_id"])
        latest_run_ids.add(latest_run)
        summary_rows.append(
            {
                **{c: v for c, v in zip(family_cols, keys if isinstance(keys, tuple) else (keys,))},
                "status": "latest_family_run_selected",
                "selected_run_id": latest_run,
                "selected_timestamp": run_summary.iloc[0].get("timestamp", ""),
                "candidate_run_count": int(len(run_summary)),
                "candidate_hybrid_rows": int(run_summary["hybrid_rows"].sum()),
            }
        )

    # A run ID is kept if it's the latest for AT LEAST ONE context it appears in.
    keep = (~hybrid_mask) | out["run_id"].astype(str).isin(latest_run_ids)
    excluded = out.loc[hybrid_mask & ~keep].copy()
    audit = pd.DataFrame(
        [
            {
                "status": "applied",
                "hybrid_rows_before": int(hybrid_mask.sum()),
                "hybrid_rows_after": int((hybrid_mask & keep).sum()),
                "stale_family_hybrid_rows_excluded": int((hybrid_mask & ~keep).sum()),
                "latest_family_runs": int(len(latest_run_ids)),
            }
        ]
        + summary_rows
    )
    if not excluded.empty:
        cols = [
            c
            for c in [
                "run_id",
                "timestamp",
                "run_modified_utc",
                "dataset",
                "model",
                "objective",
                "scope",
                "ratio",
                "variance_threshold",
                "spearman_threshold",
                "jaccard_threshold",
                "stack_id",
                "source_table",
                "source_row",
            ]
            if c in excluded.columns
        ]
        details = excluded[cols].copy()
        details.insert(0, "status", "excluded_stale_family_run")
        audit = pd.concat([audit, details], ignore_index=True, sort=False)

    return out.loc[keep].copy(), audit


def filter_hybrids_to_latest_exact_contexts(contexts: pd.DataFrame, latest_context_runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only hybrid rows from the latest run for each exact context.

    Singular rows are intentionally not filtered here because singular prunes are
    reusable by dataset/model/scope/ratio/method through singular_cache_index.csv.
    """
    if contexts.empty:
        return contexts, pd.DataFrame()
    if latest_context_runs.empty or "run_id" not in latest_context_runs.columns:
        return contexts, pd.DataFrame([{"status": "not_applied", "issue": "latest_context_runs.csv missing or empty"}])

    out = contexts.copy()
    latest = latest_context_runs.copy()
    out["_exact_context_key_for_latest_filter"] = out.apply(registry_style_context_key, axis=1)
    if "context_key" not in latest.columns:
        latest["context_key"] = latest.apply(registry_style_context_key, axis=1)

    latest_keys = latest[["context_key", "run_id"]].dropna().drop_duplicates()
    latest_pairs = set(zip(latest_keys["context_key"].astype(str), latest_keys["run_id"].astype(str)))
    hybrid_mask = out.get("record_type", pd.Series(dtype=str)).astype(str).eq("hybrid")
    pair_series = list(zip(out["_exact_context_key_for_latest_filter"].astype(str), out.get("run_id", pd.Series("", index=out.index)).astype(str)))
    keep_latest_hybrid = pd.Series([pair in latest_pairs for pair in pair_series], index=out.index)
    stale_hybrid_mask = hybrid_mask & ~keep_latest_hybrid

    audit = pd.DataFrame(
        [
            {
                "status": "applied",
                "hybrid_rows_before": int(hybrid_mask.sum()),
                "hybrid_rows_after": int((hybrid_mask & keep_latest_hybrid).sum()),
                "stale_hybrid_rows_excluded": int(stale_hybrid_mask.sum()),
                "latest_exact_context_rows": int(len(latest_keys)),
            }
        ]
    )
    stale_cols = [c for c in ["run_id", "timestamp", "dataset", "model", "objective", "scope", "ratio", "variance_threshold", "spearman_threshold", "jaccard_threshold", "stack_id", "source_table", "source_row"] if c in out.columns]
    stale_details = out.loc[stale_hybrid_mask, stale_cols].copy()
    if not stale_details.empty:
        stale_details.insert(0, "status", "excluded_stale_hybrid_context")
        audit = pd.concat([audit, stale_details], ignore_index=True, sort=False)

    filtered = out.loc[~stale_hybrid_mask].drop(columns=["_exact_context_key_for_latest_filter"], errors="ignore").copy()
    return filtered, audit


def mark_pareto(points: pd.DataFrame) -> pd.DataFrame:
    """Mark non-dominated points within one exact reporting context.

    Accuracy delta and FLOPs reduction are maximized; pruning time is minimized.
    """
    if points.empty:
        return points
    out = points.copy()
    acc = pd.to_numeric(out.get("accuracy_delta_pp", np.nan), errors="coerce").fillna(-np.inf).to_numpy()
    flops = pd.to_numeric(out.get("flops_reduction_pct", np.nan), errors="coerce").fillna(-np.inf).to_numpy()
    time = pd.to_numeric(out.get("time_sec", np.nan), errors="coerce").fillna(np.inf).to_numpy()
    is_pareto = np.ones(len(out), dtype=bool)
    for i in range(len(out)):
        for j in range(len(out)):
            if i == j:
                continue
            dominates = (
                acc[j] >= acc[i]
                and flops[j] >= flops[i]
                and time[j] <= time[i]
                and (acc[j] > acc[i] or flops[j] > flops[i] or time[j] < time[i])
            )
            if dominates:
                is_pareto[i] = False
                break
    out["is_pareto_candidate"] = is_pareto
    return out


def backfill_missing_hybrid_flops(contexts: pd.DataFrame, singular_cache: pd.DataFrame, project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backfill missing hybrid FLOPs only from checkpoint-derived v4 metrics.

    This deliberately avoids borrowing FLOPs from singular methods or from a
    different hybrid context. The only accepted backfill source is the v4 table
    produced by loading the saved hybrid checkpoint and profiling the model
    structure directly.
    """
    if contexts.empty or "flops_reduction_pct" not in contexts.columns:
        return contexts, pd.DataFrame()

    out = contexts.copy()
    out["flops_reduction_pct"] = pd.to_numeric(out["flops_reduction_pct"], errors="coerce")
    if "flops_reduction_source" not in out.columns:
        out["flops_reduction_source"] = np.where(out["flops_reduction_pct"].notna(), "artifact", "")

    audit_rows: list[dict[str, Any]] = []
    hybrid_mask = out.get("record_type", pd.Series(dtype=str)).astype(str).eq("hybrid")
    missing_mask = hybrid_mask & out["flops_reduction_pct"].isna()
    v4_path = project_root / "report_artifacts" / "context_safe_hybrid_singular_report_v4_model_metrics" / "tables" / "v4_checkpoint_direct_model_metrics.csv"
    v4 = read_csv_safe(v4_path)
    if not v4.empty:
        v4 = v4[v4.get("record_type", pd.Series(dtype=str)).astype(str).eq("hybrid")].copy()
        if "direct_flops_reduction_pct" in v4.columns:
            v4["direct_flops_reduction_pct"] = pd.to_numeric(v4["direct_flops_reduction_pct"], errors="coerce")
            v4 = v4[v4["direct_flops_reduction_pct"].notna()].copy()
        v4_keys = [c for c in ["objective", "dataset", "model", "scope", "ratio", "stack_id"] if c in out.columns and c in v4.columns]
        if v4_keys and "direct_flops_reduction_pct" in v4.columns:
            add_cols = v4_keys + [
                c
                for c in [
                    "direct_flops_reduction_pct",
                    "direct_params_reduction_pct",
                    "metric_status",
                    "checkpoint_path_resolved",
                ]
                if c in v4.columns
            ]
            v4_small = v4[add_cols].drop_duplicates(v4_keys, keep="first")
            merged = out.loc[missing_mask].merge(v4_small, on=v4_keys, how="left", suffixes=("", "_v4_direct"))
            filled = 0
            for idx, (_, row) in zip(out.loc[missing_mask].index, merged.iterrows()):
                value = safe_float(row.get("direct_flops_reduction_pct"))
                if math.isfinite(value):
                    out.at[idx, "flops_reduction_pct"] = value
                    out.at[idx, "flops_reduction_source"] = "checkpoint_direct_v4"
                    if "params_reduction_pct" in out.columns and "direct_params_reduction_pct" in row.index:
                        params_value = safe_float(row.get("direct_params_reduction_pct"))
                        if math.isfinite(params_value):
                            out.at[idx, "params_reduction_pct"] = params_value
                    filled += 1
                    audit_rows.append(
                        {
                            **{k: row.get(k) for k in COARSE_CONTEXT_KEYS},
                            "stack_id": row.get("stack_id"),
                            "source": "checkpoint_direct_v4",
                            "issue": "hybrid FLOPs backfilled from saved checkpoint direct model profile",
                            "flops_reduction_pct": value,
                            "checkpoint_path_resolved": row.get("checkpoint_path_resolved", ""),
                        }
                    )
            if filled:
                missing_mask = hybrid_mask & out["flops_reduction_pct"].isna()

    for _, row in out[missing_mask].iterrows():
        audit_rows.append(
            {
                **{k: row.get(k) for k in COARSE_CONTEXT_KEYS},
                "stack_id": row.get("stack_id"),
                "source": "missing_hybrid_flops_not_backfilled",
                "issue": "hybrid artifact has no FLOPs metric; rerun/export this exact context instead of borrowing another run",
            }
        )

    return out, pd.DataFrame(audit_rows)


def load_source_row(project_root: Path, row: pd.Series) -> pd.Series:
    source_path = resolve_path(project_root, row.get("source_table"))
    if source_path is None or not source_path.exists():
        return pd.Series(dtype=object)
    df = read_csv_safe(source_path)
    if df.empty:
        return pd.Series(dtype=object)
    idx = int(safe_float(row.get("source_row"), -1))
    if 0 <= idx < len(df):
        return df.iloc[idx]
    stack_id = norm_text(row.get("stack_id"))
    if stack_id and "stack_id" in df.columns:
        match = df[df["stack_id"].astype(str).eq(stack_id)]
        if not match.empty:
            return match.iloc[0]
    return pd.Series(dtype=object)


def rank_hybrids(contexts: pd.DataFrame, top_k: int, accuracy_gate_pp: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if contexts.empty:
        return pd.DataFrame(), pd.DataFrame()
    hybrids = contexts[contexts["record_type"].astype(str).eq("hybrid")].copy()
    if hybrids.empty:
        return pd.DataFrame(), pd.DataFrame()
    for col in [
        "ratio",
        "variance_threshold",
        "spearman_threshold",
        "jaccard_threshold",
        "accuracy_pct",
        "baseline_accuracy_pct",
        "accuracy_delta_pp",
        "flops_reduction_pct",
        "params_reduction_pct",
        "time_sec",
    ]:
        if col in hybrids.columns:
            hybrids[col] = pd.to_numeric(hybrids[col], errors="coerce")
    hybrids["_timestamp_sort"] = hybrids.get("timestamp", "").fillna("").astype(str)
    hybrids["_modified_sort"] = hybrids.get("run_modified_utc", "").fillna("").astype(str)
    dedupe_cols = [c for c in EXACT_CONTEXT_KEYS + ["stack_id", "method_or_stack"] if c in hybrids.columns]
    hybrids = (
        hybrids.sort_values(dedupe_cols + ["_timestamp_sort", "_modified_sort"], ascending=[True] * len(dedupe_cols) + [False, False])
        .drop_duplicates(dedupe_cols, keep="first")
        .copy()
    )
    groups = []
    for _, group in hybrids.groupby(COARSE_CONTEXT_KEYS, dropna=False):
        scored = score_group(group, accuracy_gate_pp)
        scored = objective_champion_order(scored, top_k=top_k).copy()
        scored["context_rank"] = np.arange(1, len(scored) + 1)
        scored["report_stack_id"] = scored.apply(stack_display_id, axis=1)
        groups.append(scored)
    ranked = pd.concat(groups, ignore_index=True) if groups else pd.DataFrame()
    ranked = assign_unique_four_digit_stack_ids(ranked)
    top = ranked[ranked["context_rank"] <= int(top_k)].copy() if not ranked.empty else pd.DataFrame()
    return ranked, top


def build_objective_selection_diagnostics(ranked: pd.DataFrame) -> pd.DataFrame:
    """Explain why each reported hybrid stack received its context rank.

    This table is deliberately redundant with the plots: it makes it obvious
    when a time objective selected a stack because it retained accuracy, and
    whether the minimum-time stack in that exact context was still slow or
    failed the accuracy gate.
    """
    if ranked.empty:
        return pd.DataFrame()
    work = ranked.copy()
    for col in ["accuracy_delta_pp", "accuracy_pct", "flops_reduction_pct", "time_sec", "context_rank"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    rows: list[dict[str, Any]] = []
    for key, group in work.groupby(COARSE_CONTEXT_KEYS, dropna=False):
        group = group.copy()
        gated = group[group.get("accuracy_gate_passed", pd.Series(False, index=group.index)).astype(bool)].copy()
        eligible = gated if not gated.empty else group
        fastest = eligible.sort_values(["time_sec", "accuracy_delta_pp", "flops_reduction_pct"], ascending=[True, False, False], na_position="last").head(1)
        acc_best = eligible.sort_values(["accuracy_delta_pp", "time_sec", "flops_reduction_pct"], ascending=[False, True, False], na_position="last").head(1)
        flops_best = eligible.sort_values(["flops_reduction_pct", "accuracy_delta_pp", "time_sec"], ascending=[False, False, True], na_position="last").head(1)

        fastest_row = fastest.iloc[0] if not fastest.empty else pd.Series(dtype=object)
        acc_row = acc_best.iloc[0] if not acc_best.empty else pd.Series(dtype=object)
        flops_row = flops_best.iloc[0] if not flops_best.empty else pd.Series(dtype=object)
        for _, row in group.sort_values("context_rank", na_position="last").iterrows():
            rows.append(
                {
                    **{k: row.get(k) for k in COARSE_CONTEXT_KEYS},
                    "objective_label": row.get("objective_label"),
                    "context_rank": row.get("context_rank"),
                    "report_stack_id": row.get("report_stack_id"),
                    "stack_id": row.get("stack_id"),
                    "rank_selection_reason": row.get("rank_selection_reason", ""),
                    "accuracy_gate_passed": bool(row.get("accuracy_gate_passed", False)),
                    "candidate_count_in_context": int(len(group)),
                    "accuracy_gate_pass_count": int(len(gated)),
                    "accuracy_delta_pp": row.get("accuracy_delta_pp"),
                    "flops_reduction_pct": row.get("flops_reduction_pct"),
                    "time_sec": row.get("time_sec"),
                    "fastest_eligible_report_stack_id": fastest_row.get("report_stack_id", ""),
                    "fastest_eligible_time_sec": fastest_row.get("time_sec", math.nan),
                    "fastest_eligible_accuracy_delta_pp": fastest_row.get("accuracy_delta_pp", math.nan),
                    "max_accuracy_report_stack_id": acc_row.get("report_stack_id", ""),
                    "max_accuracy_delta_pp": acc_row.get("accuracy_delta_pp", math.nan),
                    "max_accuracy_time_sec": acc_row.get("time_sec", math.nan),
                    "max_flops_report_stack_id": flops_row.get("report_stack_id", ""),
                    "max_flops_reduction_pct": flops_row.get("flops_reduction_pct", math.nan),
                    "max_flops_time_sec": flops_row.get("time_sec", math.nan),
                    "selection_warning": (
                        "no stack passed accuracy gate; ranks are fallback candidates"
                        if gated.empty
                        else "fastest accuracy-gate-passing stack is still slow"
                        if objective_terms(row.get("objective")) == {"time", "accuracy"} and safe_float(fastest_row.get("time_sec")) > 300
                        else ""
                    ),
                }
            )
    return pd.DataFrame(rows)


def extract_layerwise_policies(project_root: Path, top_hybrids: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for _, hybrid in top_hybrids.iterrows():
        detail = load_source_row(project_root, hybrid)
        policy = parse_literal(detail.get("layer_policy", hybrid.get("layer_policy", {})), default={})
        ratio_policy = parse_literal(detail.get("layer_ratio_policy", {}), default={})
        if not isinstance(policy, dict) or not policy:
            audits.append({**{k: hybrid.get(k) for k in COARSE_CONTEXT_KEYS}, "stack_id": hybrid.get("stack_id"), "report_stack_id": hybrid.get("report_stack_id"), "issue": "missing_layer_policy"})
            continue
        regions = assign_regions(list(policy.keys()))
        for layer_index, layer in enumerate(sorted(policy.keys(), key=layer_sort_key), start=1):
            method = policy[layer]
            rows.append(
                {
                    **{k: hybrid.get(k) for k in COARSE_CONTEXT_KEYS},
                    "objective_label": hybrid.get("objective_label"),
                    "variance_threshold": hybrid.get("variance_threshold"),
                    "spearman_threshold": hybrid.get("spearman_threshold"),
                    "jaccard_threshold": hybrid.get("jaccard_threshold"),
                    "context_rank": hybrid.get("context_rank"),
                    "report_stack_id": hybrid.get("report_stack_id"),
                    "stack_id": hybrid.get("stack_id"),
                    "layer_index": layer_index,
                    "layer_name": str(layer),
                    "region": regions.get(str(layer), "Unknown"),
                    "selected_method": method,
                    "selected_method_display": method_display(method),
                    "layer_prune_ratio": safe_float(ratio_policy.get(layer, hybrid.get("ratio")) if isinstance(ratio_policy, dict) else hybrid.get("ratio")),
                    "accuracy_delta_pp": hybrid.get("accuracy_delta_pp"),
                    "accuracy_pct": hybrid.get("accuracy_pct"),
                    "baseline_accuracy_pct": hybrid.get("baseline_accuracy_pct"),
                    "accuracy_source_column": hybrid.get("accuracy_source_column"),
                    "baseline_accuracy_source_column": hybrid.get("baseline_accuracy_source_column"),
                    "accuracy_delta_source_column": hybrid.get("accuracy_delta_source_column"),
                    "flops_reduction_pct": hybrid.get("flops_reduction_pct"),
                    "time_sec": hybrid.get("time_sec"),
                    "analysis_score": hybrid.get("analysis_score"),
                    "accuracy_gate_passed": hybrid.get("accuracy_gate_passed"),
                    "run_id": hybrid.get("run_id"),
                    "run_dir": hybrid.get("run_dir"),
                }
            )
    layer_df = pd.DataFrame(rows)
    audit_df = pd.DataFrame(audits)
    if layer_df.empty:
        return layer_df, audit_df, pd.DataFrame()
    compact = (
        layer_df.groupby(COARSE_CONTEXT_KEYS + ["context_rank", "report_stack_id", "stack_id"], dropna=False)
        .agg(
            n_layers=("layer_name", "count"),
            methods_used=("selected_method_display", lambda s: " + ".join(pd.Series(s).dropna().drop_duplicates().astype(str))),
            early_methods=("selected_method_display", lambda s: " + ".join(pd.Series(layer_df.loc[s.index][layer_df.loc[s.index, "region"].eq("Early")]["selected_method_display"]).drop_duplicates().astype(str))),
            middle_methods=("selected_method_display", lambda s: " + ".join(pd.Series(layer_df.loc[s.index][layer_df.loc[s.index, "region"].eq("Middle")]["selected_method_display"]).drop_duplicates().astype(str))),
            late_methods=("selected_method_display", lambda s: " + ".join(pd.Series(layer_df.loc[s.index][layer_df.loc[s.index, "region"].eq("Late")]["selected_method_display"]).drop_duplicates().astype(str))),
            layer_policy_text=("layer_name", lambda s: " | ".join(f"{l}: {m}" for l, m in zip(s, layer_df.loc[s.index, "selected_method_display"]))),
            accuracy_delta_pp=("accuracy_delta_pp", "first"),
            flops_reduction_pct=("flops_reduction_pct", "first"),
            time_sec=("time_sec", "first"),
            analysis_score=("analysis_score", "first"),
        )
        .reset_index()
    )
    return layer_df, audit_df, compact


def build_comparisons(top_hybrids: pd.DataFrame, singular_cache: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    if top_hybrids.empty:
        return pd.DataFrame(), pd.DataFrame()
    singular = singular_cache.copy()
    for col in ["ratio", "accuracy_pct", "baseline_accuracy_pct", "accuracy_delta_pp", "flops_reduction_pct", "time_sec"]:
        if col in singular.columns:
            singular[col] = pd.to_numeric(singular[col], errors="coerce")
    for _, hybrid in top_hybrids.iterrows():
        match = pd.DataFrame()
        if not singular.empty:
            match = singular[
                singular.get("dataset", pd.Series(dtype=str)).astype(str).eq(str(hybrid.get("dataset")))
                & singular.get("model", pd.Series(dtype=str)).astype(str).eq(str(hybrid.get("model")))
                & np.isclose(pd.to_numeric(singular.get("ratio", np.nan), errors="coerce"), safe_float(hybrid.get("ratio")))
            ].copy()
            if not match.empty and "method" in match.columns:
                match = match.sort_values(["method", "has_checkpoint_path", "timestamp", "run_modified_utc"], ascending=[True, False, False, False]).drop_duplicates("method", keep="first")
        audits.append(
            {
                **{k: hybrid.get(k) for k in COARSE_CONTEXT_KEYS},
                "context_key": context_key(hybrid),
                "objective_label": hybrid.get("objective_label"),
                "context_rank": hybrid.get("context_rank"),
                "report_stack_id": hybrid.get("report_stack_id"),
                "stack_id": hybrid.get("stack_id"),
                "singular_methods_available": int(len(match)),
                "comparison_ready": bool(len(match)),
                "issue": "ok" if len(match) else "no singular cache rows for same dataset/model/scope/ratio",
            }
        )
        for _, singular_row in match.iterrows():
            base = {
                **{k: hybrid.get(k) for k in COARSE_CONTEXT_KEYS},
                "context_key": context_key(hybrid),
                "objective_label": hybrid.get("objective_label"),
                "context_rank": hybrid.get("context_rank"),
                "report_stack_id": hybrid.get("report_stack_id"),
                "stack_id": hybrid.get("stack_id"),
                "singular_method": singular_row.get("method"),
                "singular_method_display": method_display(singular_row.get("method")),
                "singular_dataset": singular_row.get("dataset"),
                "singular_model": singular_row.get("model"),
                "singular_scope": singular_row.get("scope"),
                "singular_ratio": singular_row.get("ratio"),
                "singular_checkpoint_path": singular_row.get("checkpoint_path"),
                "singular_has_checkpoint_path": singular_row.get("has_checkpoint_path"),
                "singular_source_run_id": singular_row.get("run_id"),
                "hybrid_source_run_id": hybrid.get("run_id"),
                "hybrid_accuracy_source_column": hybrid.get("accuracy_source_column"),
                "singular_accuracy_source_column": singular_row.get("accuracy_source_column"),
                "hybrid_accuracy_delta_source_column": hybrid.get("accuracy_delta_source_column"),
                "singular_accuracy_delta_source_column": singular_row.get("accuracy_delta_source_column"),
                "context_match": (
                    str(singular_row.get("dataset")) == str(hybrid.get("dataset"))
                    and str(singular_row.get("model")) == str(hybrid.get("model"))
                    and str(singular_row.get("scope")) == str(hybrid.get("scope"))
                    and np.isclose(safe_float(singular_row.get("ratio")), safe_float(hybrid.get("ratio")))
                ),
            }
            for metric, hv, sv, higher_is_better in [
                ("accuracy_delta_pp", hybrid.get("accuracy_delta_pp"), singular_row.get("accuracy_delta_pp"), True),
                ("accuracy_pct", hybrid.get("accuracy_pct"), singular_row.get("accuracy_pct"), True),
                ("flops_reduction_pct", hybrid.get("flops_reduction_pct"), singular_row.get("flops_reduction_pct"), True),
                ("time_sec_lower_is_better", hybrid.get("time_sec"), singular_row.get("time_sec"), False),
            ]:
                hvf, svf = safe_float(hv), safe_float(sv)
                advantage = hvf - svf if higher_is_better else svf - hvf
                rows.append({**base, "metric": metric, "hybrid_value": hvf, "singular_value": svf, "hybrid_advantage_vs_singular": advantage if math.isfinite(advantage) else math.nan, "advantage_direction": "positive means hybrid is better than singular"})
    return pd.DataFrame(rows), pd.DataFrame(audits)


def augment_comparisons_with_v4_absolute_metrics(
    comparison: pd.DataFrame,
    v4_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Attach checkpoint-derived absolute compute and parameter values."""
    if comparison.empty:
        return comparison.copy()
    out = comparison.copy()
    absolute_cols = [
        "hybrid_baseline_gops",
        "hybrid_model_gops",
        "hybrid_removed_gops",
        "singular_baseline_gops",
        "singular_model_gops",
        "singular_removed_gops",
        "hybrid_baseline_params_m",
        "hybrid_model_params_m",
        "hybrid_removed_params_m",
        "singular_baseline_params_m",
        "singular_model_params_m",
        "singular_removed_params_m",
        "hybrid_absolute_metric_provenance",
        "singular_absolute_metric_provenance",
        "operation_count_convention",
    ]
    if v4_comparison.empty:
        for col in absolute_cols:
            out[col] = "" if "provenance" in col or col == "operation_count_convention" else math.nan
        return out

    direct = v4_comparison[
        v4_comparison.get("metric", pd.Series(dtype=str)).astype(str).eq("direct_flops_reduction_pct")
    ].copy()
    join_cols = [
        "objective",
        "dataset",
        "model",
        "scope",
        "ratio",
        "context_rank",
        "report_stack_id",
        "singular_method",
    ]
    usable_join_cols = [c for c in join_cols if c in out.columns and c in direct.columns]
    if not usable_join_cols:
        for col in absolute_cols:
            out[col] = "" if "provenance" in col or col == "operation_count_convention" else math.nan
        return out
    for col in usable_join_cols:
        if col == "ratio":
            out[col] = pd.to_numeric(out[col], errors="coerce").round(8)
            direct[col] = pd.to_numeric(direct[col], errors="coerce").round(8)
        elif col == "context_rank":
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
            direct[col] = pd.to_numeric(direct[col], errors="coerce").astype("Int64")
        else:
            out[col] = out[col].astype(str).str.replace(r"\.0$", "", regex=True)
            direct[col] = direct[col].astype(str).str.replace(r"\.0$", "", regex=True)
    direct = direct[
        usable_join_cols + [c for c in absolute_cols if c in direct.columns]
    ].drop_duplicates(usable_join_cols, keep="first")
    out = out.merge(direct, on=usable_join_cols, how="left")
    for col in absolute_cols:
        if col not in out.columns:
            out[col] = "" if "provenance" in col or col == "operation_count_convention" else math.nan
    return out


def build_local_flops_variability_diagnostic(comparison: pd.DataFrame) -> pd.DataFrame:
    """Summarize local-scope FLOPs variability without judging sameness.

    The report must not assume that local methods should have identical FLOPs
    reductions. This diagnostic simply exposes the measured artifact values by
    exact context so suspicious uniformity or variation can be reviewed from
    the underlying data.
    """
    if comparison.empty:
        return pd.DataFrame()
    flops = comparison[
        comparison.get("metric", pd.Series(dtype=str)).astype(str).eq("flops_reduction_pct")
        & comparison.get("scope", pd.Series(dtype=str)).astype(str).eq("local")
    ].copy()
    if flops.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = [c for c in ["objective_label", "dataset", "model", "scope", "ratio", "context_rank", "report_stack_id", "stack_id"] if c in flops.columns]
    for keys, group in flops.groupby(group_cols, dropna=False):
        rec = {c: v for c, v in zip(group_cols, keys if isinstance(keys, tuple) else (keys,))}
        hvals = pd.to_numeric(group["hybrid_value"], errors="coerce").dropna()
        svals = pd.to_numeric(group["singular_value"], errors="coerce").dropna()
        method_values = {}
        if {"singular_method_display", "singular_value"}.issubset(group.columns):
            for _, row in group.iterrows():
                value = safe_float(row.get("singular_value"))
                if math.isfinite(value):
                    method_values[str(row.get("singular_method_display", row.get("singular_method", "")))] = value
        if hvals.empty or svals.empty:
            rows.append(
                {
                    **rec,
                    "diagnostic": "local_flops_variability",
                    "status": "missing_values",
                    "hybrid_flops_reduction_pct": safe_float(hvals.iloc[0]) if not hvals.empty else math.nan,
                    "singular_unique_flops_reduction_count": int(svals.nunique()) if not svals.empty else 0,
                    "singular_values_by_method_json": json.dumps(method_values, sort_keys=True),
                    "note": "Measured artifact values only; no same-FLOPs assumption is applied.",
                }
            )
            continue
        hybrid_value = float(hvals.iloc[0])
        singular_min = float(svals.min())
        singular_max = float(svals.max())
        singular_spread = singular_max - singular_min
        rows.append(
            {
                **rec,
                "diagnostic": "local_flops_variability",
                "status": "informational",
                "hybrid_flops_reduction_pct": hybrid_value,
                "singular_mean_flops_reduction_pct": float(svals.mean()),
                "singular_median_flops_reduction_pct": float(svals.median()),
                "singular_min_flops_reduction_pct": singular_min,
                "singular_max_flops_reduction_pct": singular_max,
                "singular_spread_pct": singular_spread,
                "singular_unique_flops_reduction_count": int(svals.nunique()),
                "singular_values_by_method_json": json.dumps(method_values, sort_keys=True),
                "note": "Measured artifact values only; no same-FLOPs assumption is applied.",
            }
        )
    return pd.DataFrame(rows)


def size_from_time(values: pd.Series, min_size: int = 55, max_size: int = 320) -> pd.Series:
    rt = pd.to_numeric(values, errors="coerce")
    finite = rt[np.isfinite(rt) & (rt > 0)]
    if finite.empty:
        return pd.Series((min_size + max_size) / 2, index=rt.index)
    filled = rt.fillna(finite.median()).clip(lower=max(float(finite.min()), 1e-9))
    inv = 1.0 / filled
    lo, hi = inv.min(), inv.max()
    if abs(hi - lo) < 1e-12:
        return pd.Series((min_size + max_size) / 2, index=rt.index)
    return min_size + (inv - lo) / (hi - lo) * (max_size - min_size)


def plot_context_coverage(context_summary: pd.DataFrame, fig_dir: Path) -> Path | None:
    if context_summary.empty:
        return None
    df = context_summary.copy()
    for col in ["hybrid_rows", "singular_rows"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)
    df["comparison_ready"] = (df["hybrid_rows"] > 0) & (df["singular_rows"] > 0)
    df["label"] = df.apply(lambda r: f"{r.get('objective_label', r.get('objective'))}\n{r.get('dataset')} | {r.get('model')} | {r.get('scope')} | r={safe_float(r.get('ratio')):g}", axis=1)
    df = df.sort_values(["comparison_ready", "hybrid_rows", "singular_rows"], ascending=[False, False, False]).head(45)
    fig, ax = plt.subplots(figsize=(13, max(4.5, 0.34 * len(df))))
    y = np.arange(len(df))
    ax.barh(y - 0.16, df["hybrid_rows"], height=0.32, label="Hybrid rows", color="#2563EB")
    ax.barh(y + 0.16, df["singular_rows"], height=0.32, label="Singular rows", color="#059669")
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Artifact rows indexed")
    ax.set_title("Context evidence coverage")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = fig_dir / "context_evidence_coverage.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_alignment_readiness(alignment: pd.DataFrame, fig_dir: Path) -> Path | None:
    if alignment.empty:
        return None
    df = alignment.copy()
    df["singular_methods_available"] = pd.to_numeric(df.get("singular_methods_available", 0), errors="coerce").fillna(0)
    df["label"] = df.apply(
        lambda r: (
            f"{r.get('objective_label', r.get('objective'))}\n"
            f"{r.get('dataset')} | {r.get('model')} | {r.get('scope')} | "
            f"r={safe_float(r.get('ratio')):g} | rank={safe_float(r.get('context_rank'), 0):.0f}"
        ),
        axis=1,
    )
    df = df.sort_values(["comparison_ready", "singular_methods_available"], ascending=[False, False]).head(55)
    fig, ax = plt.subplots(figsize=(13, max(4.5, 0.32 * len(df))))
    colors = np.where(df["comparison_ready"].astype(bool), "#059669", "#DC2626")
    ax.barh(np.arange(len(df)), df["singular_methods_available"], color=colors)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Same-context singular methods available")
    ax.set_title("Hybrid-singular comparison readiness")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out = fig_dir / "hybrid_singular_comparison_readiness.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_layerwise(stack: pd.Series, layer_df: pd.DataFrame, fig_dir: Path) -> Path | None:
    sub = layer_df[layer_df["report_stack_id"].astype(str).eq(str(stack.get("report_stack_id")))].copy()
    if sub.empty:
        return None
    sub = sub.sort_values("layer_index")
    methods = list(dict.fromkeys(sub["selected_method_display"].astype(str)))
    palette = [
        "#2563EB",
        "#DC2626",
        "#059669",
        "#7C3AED",
        "#EA580C",
        "#0891B2",
        "#DB2777",
        "#65A30D",
        "#9333EA",
        "#0F766E",
        "#B45309",
        "#4F46E5",
        "#BE123C",
        "#15803D",
        "#0369A1",
        "#A16207",
    ]
    method_colors = {method: palette[i % len(palette)] for i, method in enumerate(methods)}
    xs = np.arange(len(sub))

    # Thesis-facing strip plot: one colored tile per layer, with explicit
    # Early/Middle/Late partitions. The color encodes which pruning method was
    # selected for that layer.
    fig_height = 3.9 if len(sub) <= 16 else 4.4
    fig_width = max(10.5, 0.46 * len(sub) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    region_fills = {"Early": "#DBEAFE", "Middle": "#DCFCE7", "Late": "#FEF3C7"}
    for region in ["Early", "Middle", "Late"]:
        rsub = sub[sub["region"].astype(str).eq(region)]
        if rsub.empty:
            continue
        start = int(rsub["layer_index"].min()) - 1
        end = int(rsub["layer_index"].max()) - 1
        ax.axvspan(start - 0.5, end + 0.5, ymin=0.56, ymax=1.0, color=region_fills[region], alpha=0.95, zorder=0)
        ax.text((start + end) / 2, 1.18, region, ha="center", va="center", fontsize=9, fontweight="bold", color="#0F172A")
        if start > 0:
            ax.axvline(start - 0.5, color="#0F172A", linewidth=1.1, alpha=0.65, ymin=0.10, ymax=0.98)

    bars = ax.bar(
        xs,
        np.ones(len(sub)),
        bottom=0.0,
        width=0.82,
        color=[method_colors[m] for m in sub["selected_method_display"].astype(str)],
        edgecolor="#FFFFFF",
        linewidth=0.8,
        zorder=2,
    )

    for rect, (_, row) in zip(bars, sub.iterrows()):
        method = str(row.get("selected_method_display", ""))
        label = method if len(method) <= 9 else method[:8] + "."
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            0.5,
            label,
            ha="center",
            va="center",
            rotation=90 if len(sub) > 18 else 0,
            fontsize=6.5,
            color="white",
            fontweight="bold",
            clip_on=True,
        )

    ax.set_xlim(-0.6, len(sub) - 0.4)
    ax.set_ylim(-0.38, 1.32)
    ax.set_yticks([])
    ax.set_xticks(xs)
    ax.set_xticklabels(sub["layer_name"], rotation=55, ha="right", fontsize=7.5)
    ax.set_xlabel("Prunable layer")
    ax.set_title(
        f"Layer-wise pruning policy for stack {stack.get('report_stack_id')} | rank #{int(stack.get('context_rank'))} | {stack.get('objective_label')}\n"
        f"{stack.get('dataset')} | {stack.get('model')} | {stack.get('scope')} | r={safe_float(stack.get('ratio')):g} | "
        f"Acc {safe_float(stack.get('accuracy_delta_pp')):.2f} pp, FLOPs {safe_float(stack.get('flops_reduction_pct')):.2f}%, Time {safe_float(stack.get('time_sec')):.2f}s",
        fontsize=10,
    )

    handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=method_colors[m], markersize=8, label=m)
        for m in methods
    ]
    legend_cols = min(6, max(1, len(methods)))
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.46),
        ncol=legend_cols,
        fontsize=7,
        frameon=False,
        title="Selected pruning method",
        title_fontsize=8,
    )
    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#94A3B8")
    fig.tight_layout()
    out = fig_dir / f"layerwise_{safe_slug(stack.get('objective'))}_{safe_slug(stack.get('dataset'))}_{safe_slug(stack.get('model'))}_{safe_slug(stack.get('scope'))}_r{safe_slug(stack.get('ratio'))}_rank{int(stack.get('context_rank'))}_{safe_slug(stack.get('report_stack_id'))}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_comparison(stack: pd.Series, comparison: pd.DataFrame, fig_dir: Path) -> Path | None:
    comp = comparison[
        comparison["report_stack_id"].astype(str).eq(str(stack.get("report_stack_id")))
        & comparison["objective"].astype(str).eq(str(stack.get("objective")))
        & comparison["dataset"].astype(str).eq(str(stack.get("dataset")))
        & comparison["model"].astype(str).eq(str(stack.get("model")))
        & comparison["scope"].astype(str).eq(str(stack.get("scope")))
        & np.isclose(pd.to_numeric(comparison["ratio"], errors="coerce"), safe_float(stack.get("ratio")))
    ].copy()
    if comp.empty:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    specs = [
        ("accuracy_delta_pp", "Test accuracy delta vs baseline (pp)", "Hybrid test accuracy delta", "#10B981", False),
        ("flops_reduction_pct", "Structural FLOPs reduction (%)", "Hybrid structural FLOPs reduction", "#2563EB", False),
        ("time_sec_lower_is_better", "Pruning time (s)", "Hybrid pruning time", "#F97316", True),
    ]
    for ax, (metric, ylabel, hlabel, color, lower_is_better) in zip(axes, specs):
        sub = comp[comp["metric"].eq(metric)].dropna(subset=["singular_value"]).copy()
        sub = sub.sort_values("singular_value", ascending=lower_is_better)
        hv = safe_float(sub["hybrid_value"].dropna().iloc[0]) if not sub["hybrid_value"].dropna().empty else math.nan
        if not sub.empty:
            bars = ax.bar(sub["singular_method_display"], sub["singular_value"], color=color, alpha=0.72, label="Singular method")
            try:
                if metric == "flops_reduction_pct" and "singular_model_gops" in sub.columns:
                    labels = [
                        f"{value:.1f}%\n{safe_float(gops):.3f}"
                        if math.isfinite(safe_float(gops))
                        else f"{value:.2f}%"
                        for value, gops in zip(sub["singular_value"], sub["singular_model_gops"])
                    ]
                else:
                    labels = [f"{v:.2f}" for v in sub["singular_value"]]
                ax.bar_label(bars, labels=labels, padding=2, fontsize=7)
            except Exception:
                pass
        else:
            ax.text(0.5, 0.5, "No same-context singular values", transform=ax.transAxes, ha="center", va="center")
        if math.isfinite(hv):
            hybrid_label = hlabel
            if metric == "flops_reduction_pct" and "hybrid_model_gops" in sub.columns:
                hybrid_gops = safe_float(sub["hybrid_model_gops"].dropna().iloc[0]) if not sub["hybrid_model_gops"].dropna().empty else math.nan
                if math.isfinite(hybrid_gops):
                    hybrid_label = f"{hlabel}: {hv:.2f}% / {hybrid_gops:.3f} GOp remaining"
                baseline_gops = safe_float(sub["hybrid_baseline_gops"].dropna().iloc[0]) if "hybrid_baseline_gops" in sub.columns and not sub["hybrid_baseline_gops"].dropna().empty else math.nan
                if math.isfinite(baseline_gops):
                    ax.text(
                        0.02,
                        0.06,
                        f"Unpruned baseline: {baseline_gops:.3f} GOp",
                        transform=ax.transAxes,
                        ha="left",
                        va="bottom",
                        fontsize=8,
                        color="#334155",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#CBD5E1", alpha=0.88),
                    )
            ax.axhline(hv, color="#111827", linestyle="--", linewidth=1.5, label=hybrid_label)
        else:
            ax.text(
                0.5,
                0.93,
                "Hybrid value missing in source artifact",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=8,
                color="#991B1B",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#FEF2F2", edgecolor="#FCA5A5", alpha=0.92),
            )
        if metric == "accuracy_delta_pp":
            ax.axhline(0, color="#64748B", linewidth=0.8)
        if metric == "flops_reduction_pct":
            values = pd.to_numeric(sub.get("singular_value", pd.Series(dtype=float)), errors="coerce").dropna()
            ymax = max([safe_float(hv, 0.0), *values.tolist(), 1.0])
            ax.set_ylim(top=ymax * 1.18)
            if "singular_model_gops" in sub.columns and sub["singular_model_gops"].notna().any():
                ax.text(
                    0.98,
                    0.06,
                    "Bar labels: reduction % / remaining GOp",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=7,
                    color="#475569",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#CBD5E1", alpha=0.88),
                )
        ax.set_ylabel(ylabel)
        if metric == "flops_reduction_pct":
            ax.set_title("FLOPs reduction and remaining compute", pad=46)
        else:
            ax.set_title(metric.replace("_", " "))
        ax.tick_params(axis="x", rotation=55)
        ax.grid(axis="y", alpha=0.25)
        if metric == "flops_reduction_pct":
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=6.5)
        else:
            ax.legend(fontsize=7)
    gate_label = "accuracy gate pass" if bool(stack.get("accuracy_gate_passed", False)) else "accuracy gate fail"
    reason = norm_text(stack.get("rank_selection_reason")) or "ranked"
    fig.suptitle(
        f"Hybrid vs same-context singular methods | {stack.get('report_stack_id')} | {stack.get('objective_label')} | {stack.get('dataset')} | {stack.get('model')} | {stack.get('scope')} | r={safe_float(stack.get('ratio')):g}\n"
        f"rank #{int(safe_float(stack.get('context_rank'), 0))}: {reason}; {gate_label}; "
        f"acc {safe_float(stack.get('accuracy_delta_pp')):.2f} pp, FLOPs {safe_float(stack.get('flops_reduction_pct')):.2f}%, time {safe_float(stack.get('time_sec')):.2f}s",
        y=1.08,
        fontsize=10,
    )
    fig.tight_layout()
    out = fig_dir / f"comparison_{safe_slug(stack.get('objective'))}_{safe_slug(stack.get('dataset'))}_{safe_slug(stack.get('model'))}_{safe_slug(stack.get('scope'))}_r{safe_slug(stack.get('ratio'))}_rank{int(stack.get('context_rank'))}_{safe_slug(stack.get('report_stack_id'))}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_absolute_footprint(stack: pd.Series, comparison: pd.DataFrame, fig_dir: Path) -> Path | None:
    """Create an appendix-ready absolute remaining GOp and parameter plot."""
    comp = comparison[
        comparison["report_stack_id"].astype(str).eq(str(stack.get("report_stack_id")))
        & comparison["objective"].astype(str).eq(str(stack.get("objective")))
        & comparison["dataset"].astype(str).eq(str(stack.get("dataset")))
        & comparison["model"].astype(str).eq(str(stack.get("model")))
        & comparison["scope"].astype(str).eq(str(stack.get("scope")))
        & np.isclose(pd.to_numeric(comparison["ratio"], errors="coerce"), safe_float(stack.get("ratio")))
    ].copy()
    if comp.empty or not {"singular_model_gops", "singular_model_params_m"}.issubset(comp.columns):
        return None

    method_rows = comp.sort_values("singular_method_display").drop_duplicates("singular_method_display")
    if method_rows["singular_model_gops"].notna().sum() == 0 and method_rows["singular_model_params_m"].notna().sum() == 0:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    panels = [
        ("singular_model_gops", "hybrid_model_gops", "hybrid_baseline_gops", "Remaining compute (GOp)", "#2563EB"),
        ("singular_model_params_m", "hybrid_model_params_m", "hybrid_baseline_params_m", "Remaining parameters (millions)", "#7C3AED"),
    ]
    for ax, (singular_col, hybrid_col, baseline_col, ylabel, color) in zip(axes, panels):
        panel = method_rows.dropna(subset=[singular_col]).sort_values(singular_col)
        if not panel.empty:
            bars = ax.bar(panel["singular_method_display"], panel[singular_col], color=color, alpha=0.72, label="Singular method")
            ax.bar_label(bars, labels=[f"{v:.3f}" for v in panel[singular_col]], padding=2, fontsize=7)
        hybrid_value = safe_float(method_rows[hybrid_col].dropna().iloc[0]) if not method_rows[hybrid_col].dropna().empty else math.nan
        baseline_value = safe_float(method_rows[baseline_col].dropna().iloc[0]) if not method_rows[baseline_col].dropna().empty else math.nan
        if math.isfinite(hybrid_value):
            ax.axhline(hybrid_value, color="#111827", linestyle="--", linewidth=1.6, label=f"Hybrid: {hybrid_value:.3f}")
        if math.isfinite(baseline_value):
            ax.axhline(baseline_value, color="#64748B", linestyle=":", linewidth=1.2, label=f"Unpruned baseline: {baseline_value:.3f}")
        ax.set_title(ylabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=55)
        ax.grid(axis="y", alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=8)
    fig.suptitle(
        f"Absolute checkpoint-derived footprint | {stack.get('report_stack_id')} | {stack.get('objective_label')} | "
        f"{stack.get('dataset')} | {stack.get('model')} | {stack.get('scope')} | r={safe_float(stack.get('ratio')):g}",
        fontsize=11,
    )
    fig.text(0.5, 0.01, "GOp convention: one multiply-accumulate is counted as one operation.", ha="center", fontsize=8, color="#475569")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    out = fig_dir / f"absolute_{safe_slug(stack.get('objective'))}_{safe_slug(stack.get('dataset'))}_{safe_slug(stack.get('model'))}_{safe_slug(stack.get('scope'))}_r{safe_slug(stack.get('ratio'))}_rank{int(stack.get('context_rank'))}_{safe_slug(stack.get('report_stack_id'))}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_paretos(ranked: pd.DataFrame, singular_cache: pd.DataFrame, fig_dir: Path, accuracy_gate_pp: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    point_frames = []
    if ranked.empty:
        return pd.DataFrame(), pd.DataFrame()
    for key, hgrp in ranked.groupby(COARSE_CONTEXT_KEYS, dropna=False):
        objective, dataset, model, scope, ratio = key
        singular = singular_cache.copy()
        if not singular.empty:
            singular = singular[
                singular.get("dataset", pd.Series(dtype=str)).astype(str).eq(str(dataset))
                & singular.get("model", pd.Series(dtype=str)).astype(str).eq(str(model))
                & np.isclose(pd.to_numeric(singular.get("ratio", np.nan), errors="coerce"), safe_float(ratio))
            ].copy()
            singular["report_stack_id"] = singular.get("method", pd.Series(dtype=str)).apply(method_display)
            if "method" in singular.columns:
                singular = singular.sort_values(["method", "has_checkpoint_path", "timestamp", "run_modified_utc"], ascending=[True, False, False, False]).drop_duplicates("method", keep="first")
        hybrids = hgrp.copy()
        hybrids["strategy_label"] = "Hybrid"
        singular["strategy_label"] = "Singular"
        common_cols = sorted(set(hybrids.columns).union(singular.columns))
        combo = pd.concat([hybrids.reindex(columns=common_cols), singular.reindex(columns=common_cols)], ignore_index=True)
        for col in ["accuracy_delta_pp", "flops_reduction_pct", "time_sec"]:
            if col in combo.columns:
                combo[col] = pd.to_numeric(combo[col], errors="coerce")
        combo = combo.dropna(subset=["accuracy_delta_pp", "flops_reduction_pct"])
        if combo.empty:
            continue
        combo = mark_pareto(combo)
        combo["context_key"] = "|".join(str(x) for x in [objective, dataset, model, scope, ratio])
        combo["pareto_context_objective"] = objective
        combo["pareto_context_dataset"] = dataset
        combo["pareto_context_model"] = model
        combo["pareto_context_scope"] = scope
        combo["pareto_context_ratio"] = ratio
        combo["point_label"] = combo.apply(
            lambda r: str(r.get("report_stack_id") or method_display(r.get("method"))),
            axis=1,
        )
        point_frames.append(
            combo[
                [
                    "context_key",
                    "pareto_context_objective",
                    "pareto_context_dataset",
                    "pareto_context_model",
                    "pareto_context_scope",
                    "pareto_context_ratio",
                    "strategy_label",
                    "point_label",
                    "report_stack_id",
                    "stack_id",
                    "method",
                    "accuracy_delta_pp",
                    "accuracy_pct",
                    "flops_reduction_pct",
                    "time_sec",
                    "is_pareto_candidate",
                ]
                if set(
                    [
                        "context_key",
                        "pareto_context_objective",
                        "pareto_context_dataset",
                        "pareto_context_model",
                        "pareto_context_scope",
                        "pareto_context_ratio",
                        "strategy_label",
                        "point_label",
                        "report_stack_id",
                        "stack_id",
                        "method",
                        "accuracy_delta_pp",
                        "accuracy_pct",
                        "flops_reduction_pct",
                        "time_sec",
                        "is_pareto_candidate",
                    ]
                ).issubset(combo.columns)
                else combo.columns
            ].copy()
        )
        fig, ax = plt.subplots(figsize=(9.8, 6.6))
        for label, color, marker in [("Hybrid", "#2563EB", "o"), ("Singular", "#F97316", "s")]:
            sub = combo[combo["strategy_label"].eq(label)].copy()
            if sub.empty:
                continue
            sizes = size_from_time(sub["time_sec"])
            ax.scatter(sub["flops_reduction_pct"], sub["accuracy_delta_pp"], s=sizes, alpha=0.68, label=f"{label} (size=faster time)", color=color, marker=marker, edgecolor="#111827", linewidth=0.5)
            for _, point in sub.iterrows():
                label_text = str(point.get("point_label"))
                offset = (6, 6) if label == "Hybrid" else (5, -11)
                ax.annotate(
                    label_text,
                    (point["flops_reduction_pct"], point["accuracy_delta_pp"]),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#CBD5E1", alpha=0.78),
                    zorder=6,
                )
        pareto = combo[combo["is_pareto_candidate"].astype(bool)].copy()
        if not pareto.empty:
            pareto_sizes = size_from_time(pareto["time_sec"], min_size=180, max_size=520)
            ax.scatter(
                pareto["flops_reduction_pct"],
                pareto["accuracy_delta_pp"],
                s=pareto_sizes,
                marker="*",
                color="#111827",
                edgecolor="#FBBF24",
                linewidth=0.9,
                label="Pareto candidate",
                zorder=7,
            )
        xvals = pd.to_numeric(combo["flops_reduction_pct"], errors="coerce").dropna()
        yvals = pd.to_numeric(combo["accuracy_delta_pp"], errors="coerce").dropna()
        if not xvals.empty:
            xspan = max(float(xvals.max() - xvals.min()), 2.0)
            ax.set_xlim(max(0.0, float(xvals.min()) - 0.12 * xspan - 1.0), float(xvals.max()) + 0.12 * xspan + 1.0)
        if not yvals.empty:
            yspan = max(float(yvals.max() - yvals.min()), 3.0)
            ax.set_ylim(float(yvals.min()) - 0.14 * yspan - 1.0, float(yvals.max()) + 0.14 * yspan + 1.0)
        ax.axhline(-accuracy_gate_pp, color="#DC2626", linestyle="--", lw=1.0, label=f"accuracy gate -{accuracy_gate_pp:g} pp")
        ax.axhline(0, color="#64748B", lw=0.8)
        ax.set_xlabel("FLOPs reduction (%)")
        ax.set_ylabel("Test accuracy delta vs baseline (pp)")
        ax.set_title(f"Pareto context | {objective} | {dataset} | {model} | {scope} | r={safe_float(ratio):g}\nHybrid labels match layerwise plots; singular labels are pruning methods")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = fig_dir / f"pareto_{safe_slug(objective)}_{safe_slug(dataset)}_{safe_slug(model)}_{safe_slug(scope)}_r{safe_slug(ratio)}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        rows.append({
            "objective": objective,
            "dataset": dataset,
            "model": model,
            "scope": scope,
            "ratio": ratio,
            "plot": str(out),
            "hybrid_points": int((combo["strategy_label"] == "Hybrid").sum()),
            "singular_points": int((combo["strategy_label"] == "Singular").sum()),
            "pareto_candidates": int(combo["is_pareto_candidate"].sum()),
            "hybrid_pareto_candidates": int(((combo["strategy_label"] == "Hybrid") & combo["is_pareto_candidate"]).sum()),
            "singular_pareto_candidates": int(((combo["strategy_label"] == "Singular") & combo["is_pareto_candidate"]).sum()),
        })
    pareto_points = pd.concat(point_frames, ignore_index=True) if point_frames else pd.DataFrame()
    return pd.DataFrame(rows), pareto_points


def plot_region_heatmaps(layer_df: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    rows = []
    if layer_df.empty:
        return pd.DataFrame()
    for key, group in layer_df.groupby(["objective_label", "dataset", "model", "scope", "ratio"], dropna=False):
        objective_label, dataset, model, scope, ratio = key
        pivot = group.pivot_table(index="region", columns="selected_method_display", values="layer_name", aggfunc="count", fill_value=0)
        pivot = pivot.reindex(index=["Early", "Middle", "Late"]).fillna(0)
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(pivot.columns) + 3), 3.9))
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = int(pivot.iloc[i, j])
                if value:
                    ax.text(j, i, str(value), ha="center", va="center", fontsize=8)
        ax.set_title(f"Layer-region method choices in hybrid stacks\n{objective_label} | {dataset} | {model} | {scope} | r={safe_float(ratio):g}")
        ax.set_xlabel("Selected pruning method")
        ax.set_ylabel("Layer region")
        fig.colorbar(im, ax=ax, label="Layer assignments in selected top stacks")
        fig.tight_layout()
        out = fig_dir / f"region_method_heatmap_{safe_slug(objective_label)}_{safe_slug(dataset)}_{safe_slug(model)}_{safe_slug(scope)}_r{safe_slug(ratio)}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        rows.append({"objective_label": objective_label, "dataset": dataset, "model": model, "scope": scope, "ratio": ratio, "plot": str(out)})
    return pd.DataFrame(rows)


def build_top_rank_coverage(ranked: pd.DataFrame, top: pd.DataFrame, required_top_k: int = 2) -> pd.DataFrame:
    if ranked.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    top_ids = set()
    if not top.empty and {"report_stack_id", "context_rank"}.issubset(top.columns):
        top_ids = set(
            zip(
                top["report_stack_id"].astype(str),
                pd.to_numeric(top["context_rank"], errors="coerce").fillna(-1).astype(int),
            )
        )
    for key, group in ranked.groupby(COARSE_CONTEXT_KEYS, dropna=False):
        group = group.copy()
        ranks = sorted(pd.to_numeric(group.get("context_rank", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).unique())
        available_top_ranks = [r for r in ranks if r <= int(required_top_k)]
        missing_required = [r for r in range(1, int(required_top_k) + 1) if r not in available_top_ranks]
        selected = group[pd.to_numeric(group.get("context_rank", np.nan), errors="coerce").isin(available_top_ranks)].copy()
        plot_ready = []
        for _, row in selected.iterrows():
            pair = (str(row.get("report_stack_id")), int(safe_float(row.get("context_rank"), -1)))
            plot_ready.append(pair in top_ids)
        rows.append(
            {
                **{k: v for k, v in zip(COARSE_CONTEXT_KEYS, key)},
                "ranked_hybrid_count": int(len(group)),
                "required_top_k": int(required_top_k),
                "available_top_rank_count": int(len(available_top_ranks)),
                "available_top_ranks": " + ".join(str(r) for r in available_top_ranks),
                "missing_required_ranks": " + ".join(str(r) for r in missing_required),
                "rank1_available": 1 in available_top_ranks,
                "rank2_available": 2 in available_top_ranks,
                "rank1_and_rank2_available": all(r in available_top_ranks for r in [1, 2]),
                "selected_top_stack_ids": " + ".join(selected.sort_values("context_rank")["report_stack_id"].astype(str)) if "report_stack_id" in selected.columns and not selected.empty else "",
                "selected_top_stack_full_ids": " + ".join(selected.sort_values("context_rank")["stack_id"].astype(str)) if "stack_id" in selected.columns and not selected.empty else "",
                "all_selected_top_plots_ready": bool(all(plot_ready)) if plot_ready else False,
            }
        )
    return pd.DataFrame(rows)


def write_outputs(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    outputs_root = (project_root / args.outputs_root).resolve() if not args.outputs_root.is_absolute() else args.outputs_root
    registry_dir = (project_root / args.registry_dir).resolve() if not args.registry_dir.is_absolute() else args.registry_dir
    report_dir = (project_root / args.report_dir).resolve() if not args.report_dir.is_absolute() else args.report_dir
    table_dir = report_dir / "tables"
    fig_dir = report_dir / "plots"
    layer_fig_dir = fig_dir / "layerwise_policies"
    comp_fig_dir = fig_dir / "hybrid_vs_singular"
    absolute_fig_dir = fig_dir / "absolute_footprints"
    pareto_fig_dir = fig_dir / "pareto"
    summary_fig_dir = fig_dir / "summary"
    generated_dirs = [table_dir] if args.skip_plots else [table_dir, fig_dir]
    for generated_dir in generated_dirs:
        if generated_dir.exists():
            shutil.rmtree(generated_dir, ignore_errors=True)
    for directory in [table_dir, layer_fig_dir, comp_fig_dir, absolute_fig_dir, pareto_fig_dir, summary_fig_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    if args.refresh_registry:
        subprocess.run(
            [
                sys.executable,
                str(project_root / "scripts" / "build_experiment_registry.py"),
                "--outputs-root",
                str(outputs_root),
                "--registry-dir",
                str(registry_dir),
                "--accuracy-gate-pp",
                str(args.accuracy_gate_pp),
                "--top-k",
                str(max(args.top_k, 2)),
            ],
            check=True,
        )

    contexts = read_csv_safe(registry_dir / "contexts.csv")
    singular_cache = read_csv_safe(registry_dir / "singular_cache_index.csv")
    latest_context_runs = read_csv_safe(registry_dir / "latest_context_runs.csv")
    allowed_ratios = parse_allowed_ratios(args.allowed_ratios)
    contexts, allowed_context_ratio_audit = filter_to_allowed_ratios(contexts, allowed_ratios, "contexts")
    singular_cache, allowed_singular_ratio_audit = filter_to_allowed_ratios(singular_cache, allowed_ratios, "singular_cache_index")
    latest_context_runs, allowed_latest_ratio_audit = filter_to_allowed_ratios(latest_context_runs, allowed_ratios, "latest_context_runs")
    contexts, latest_family_run_filter_audit = filter_hybrids_to_latest_family_runs(contexts)
    contexts, latest_context_filter_audit = filter_hybrids_to_latest_exact_contexts(contexts, latest_context_runs)
    contexts, flops_backfill_audit = backfill_missing_hybrid_flops(contexts, singular_cache, project_root)
    context_summary = read_csv_safe(registry_dir / "context_summary.csv")
    context_summary, allowed_context_summary_ratio_audit = filter_to_allowed_ratios(context_summary, allowed_ratios, "context_summary")
    registry_quality_audit = read_csv_safe(registry_dir / "registry_quality_audit.csv")

    allowed_ratio_filter_audit = pd.concat(
        [
            x
            for x in [
                allowed_context_ratio_audit,
                allowed_singular_ratio_audit,
                allowed_latest_ratio_audit,
                allowed_context_summary_ratio_audit,
            ]
            if x is not None and not x.empty
        ],
        ignore_index=True,
    ) if allowed_ratios else pd.DataFrame()
    allowed_ratio_filter_audit.to_csv(table_dir / "allowed_prune_ratio_filter_audit.csv", index=False)
    latest_family_run_filter_audit.to_csv(table_dir / "latest_family_run_filter_audit.csv", index=False)
    latest_context_filter_audit.to_csv(table_dir / "latest_exact_context_filter_audit.csv", index=False)
    flops_backfill_audit.to_csv(table_dir / "hybrid_flops_backfill_audit.csv", index=False)
    context_summary.to_csv(table_dir / "context_overview.csv", index=False)
    registry_quality_audit.to_csv(table_dir / "registry_quality_audit.csv", index=False)
    coverage_plot = None if args.skip_plots else plot_context_coverage(context_summary, summary_fig_dir)

    ranked, top = rank_hybrids(contexts, args.top_k, args.accuracy_gate_pp)
    ranked.to_csv(table_dir / "all_ranked_hybrid_stacks_by_context.csv", index=False)
    top.to_csv(table_dir / "top_hybrid_stacks_by_context.csv", index=False)
    selection_diagnostics = build_objective_selection_diagnostics(ranked)
    selection_diagnostics.to_csv(table_dir / "objective_selection_diagnostics.csv", index=False)
    top_rank_coverage = build_top_rank_coverage(ranked, top, required_top_k=args.top_k)
    top_rank_coverage.to_csv(table_dir / "top_hybrid_policy_coverage_audit.csv", index=False)
    top_rank_coverage.to_csv(table_dir / "top2_hybrid_policy_coverage_audit.csv", index=False)  # backward-compatible alias

    layerwise, policy_audit, compact_policy = extract_layerwise_policies(project_root, ranked)
    layerwise.to_csv(table_dir / "hybrid_layerwise_policy_linked_to_metrics.csv", index=False)
    policy_audit.to_csv(table_dir / "hybrid_layerwise_policy_audit.csv", index=False)
    compact_policy.to_csv(table_dir / "hybrid_compact_layerwise_policies.csv", index=False)

    comparison, alignment = build_comparisons(top, singular_cache)
    v4_report_dir = (project_root / args.v4_report_dir).resolve() if not args.v4_report_dir.is_absolute() else args.v4_report_dir
    v4_table_dir = v4_report_dir / "tables"
    v4_comparison = read_csv_safe(v4_table_dir / "v4_hybrid_vs_singular_checkpoint_direct_long.csv")
    comparison = augment_comparisons_with_v4_absolute_metrics(comparison, v4_comparison)
    comparison.to_csv(table_dir / "hybrid_vs_singular_exact_context_long.csv", index=False)
    baseline_scale = read_csv_safe(v4_table_dir / "v4_baseline_model_scale.csv")
    baseline_scale.to_csv(table_dir / "baseline_model_scale.csv", index=False)
    alignment.to_csv(table_dir / "hybrid_singular_alignment_audit.csv", index=False)
    local_flops_variability = build_local_flops_variability_diagnostic(comparison)
    local_flops_variability.to_csv(table_dir / "local_scope_flops_variability_diagnostic.csv", index=False)
    readiness_plot = None if args.skip_plots else plot_alignment_readiness(alignment, summary_fig_dir)

    selected_for_plots = pd.DataFrame() if args.skip_plots else (top.head(args.max_plots) if args.max_plots is not None else top)
    plot_rows = []
    for _, stack in selected_for_plots.iterrows():
        layer_plot = plot_layerwise(stack, layerwise, layer_fig_dir)
        comparison_plot = plot_comparison(stack, comparison, comp_fig_dir)
        absolute_plot = plot_absolute_footprint(stack, comparison, absolute_fig_dir)
        plot_rows.append(
            {
                **{k: stack.get(k) for k in COARSE_CONTEXT_KEYS},
                "objective_label": stack.get("objective_label"),
                "context_rank": stack.get("context_rank"),
                "report_stack_id": stack.get("report_stack_id"),
                "stack_id": stack.get("stack_id"),
                "layerwise_plot": str(layer_plot) if layer_plot else "",
                "comparison_plot": str(comparison_plot) if comparison_plot else "",
                "absolute_footprint_plot": str(absolute_plot) if absolute_plot else "",
            }
        )
    plot_manifest = pd.DataFrame(plot_rows)
    plot_manifest.to_csv(table_dir / "plot_manifest_layerwise_and_comparison.csv", index=False)
    if not top_rank_coverage.empty and not plot_manifest.empty:
        plot_counts = (
            plot_manifest.assign(
                has_layerwise_plot=plot_manifest["layerwise_plot"].astype(str).ne(""),
                has_comparison_plot=plot_manifest["comparison_plot"].astype(str).ne(""),
            )
            .groupby(COARSE_CONTEXT_KEYS, dropna=False)
            .agg(
                selected_plot_rows=("report_stack_id", "count"),
                selected_layerwise_plots=("has_layerwise_plot", "sum"),
                selected_comparison_plots=("has_comparison_plot", "sum"),
                plotted_ranks=("context_rank", lambda s: " + ".join(str(int(x)) for x in sorted(pd.to_numeric(s, errors="coerce").dropna().astype(int).unique()))),
                plotted_stack_ids=("report_stack_id", lambda s: " + ".join(pd.Series(s).dropna().astype(str))),
            )
            .reset_index()
        )
        top_rank_coverage = top_rank_coverage.drop(columns=["all_selected_top_plots_ready"], errors="ignore").merge(plot_counts, on=COARSE_CONTEXT_KEYS, how="left")
        top_rank_coverage["all_selected_top_plots_ready"] = (
            pd.to_numeric(top_rank_coverage.get("available_top_rank_count", 0), errors="coerce").fillna(0).astype(int).eq(
                pd.to_numeric(top_rank_coverage.get("selected_layerwise_plots", 0), errors="coerce").fillna(0).astype(int)
            )
            & pd.to_numeric(top_rank_coverage.get("available_top_rank_count", 0), errors="coerce").fillna(0).astype(int).eq(
                pd.to_numeric(top_rank_coverage.get("selected_comparison_plots", 0), errors="coerce").fillna(0).astype(int)
            )
        )
        top_rank_coverage.to_csv(table_dir / "top_hybrid_policy_coverage_audit.csv", index=False)
        top_rank_coverage.to_csv(table_dir / "top2_hybrid_policy_coverage_audit.csv", index=False)  # backward-compatible alias
    if args.skip_plots:
        pareto_manifest, pareto_points = pd.DataFrame(), pd.DataFrame()
    else:
        pareto_manifest, pareto_points = plot_paretos(ranked, singular_cache, pareto_fig_dir, args.accuracy_gate_pp)
    pareto_manifest.to_csv(table_dir / "plot_manifest_pareto.csv", index=False)
    pareto_points.to_csv(table_dir / "pareto_points_by_exact_context.csv", index=False)
    pareto_candidates = pareto_points[pareto_points.get("is_pareto_candidate", pd.Series(dtype=bool)).astype(bool)].copy() if not pareto_points.empty else pd.DataFrame()
    pareto_candidates.to_csv(table_dir / "pareto_candidates_by_exact_context.csv", index=False)
    region_manifest = pd.DataFrame() if args.skip_plots else plot_region_heatmaps(layerwise, summary_fig_dir)
    region_manifest.to_csv(table_dir / "plot_manifest_region_method_heatmaps.csv", index=False)

    if not top.empty:
        best_policy_report = top.merge(
            compact_policy[
                [
                    "report_stack_id",
                    "methods_used",
                    "early_methods",
                    "middle_methods",
                    "late_methods",
                    "layer_policy_text",
                    "n_layers",
                ]
            ]
            if not compact_policy.empty
            else pd.DataFrame(columns=["report_stack_id"]),
            on="report_stack_id",
            how="left",
        ).merge(
            alignment[["report_stack_id", "singular_methods_available", "comparison_ready", "issue"]]
            if not alignment.empty
            else pd.DataFrame(columns=["report_stack_id"]),
            on="report_stack_id",
            how="left",
        )
    else:
        best_policy_report = pd.DataFrame()
    keep_cols = [
        "objective_label",
        "dataset",
        "model",
        "scope",
        "ratio",
        "context_rank",
        "report_stack_id",
        "stack_id",
        "rank_selection_reason",
        "accuracy_gate_passed",
        "analysis_score",
        "accuracy_delta_pp",
        "accuracy_pct",
        "baseline_accuracy_pct",
        "accuracy_source_column",
        "baseline_accuracy_source_column",
        "accuracy_delta_source_column",
        "flops_reduction_pct",
        "time_sec",
        "methods_used",
        "early_methods",
        "middle_methods",
        "late_methods",
        "n_layers",
        "singular_methods_available",
        "comparison_ready",
        "issue",
        "layer_policy_text",
    ]
    if not best_policy_report.empty:
        best_policy_report = best_policy_report[[c for c in keep_cols if c in best_policy_report.columns]]
    best_policy_report.to_csv(table_dir / "integrated_top_policy_report.csv", index=False)

    if not best_policy_report.empty:
        summary = (
            best_policy_report.groupby(["objective_label", "dataset", "model", "scope", "ratio"], dropna=False)
            .agg(
                reported_stacks=("report_stack_id", "count"),
                accuracy_gate_pass_rate=("accuracy_gate_passed", "mean"),
                best_accuracy_delta_pp=("accuracy_delta_pp", "max"),
                mean_accuracy_delta_pp=("accuracy_delta_pp", "mean"),
                best_flops_reduction_pct=("flops_reduction_pct", "max"),
                mean_flops_reduction_pct=("flops_reduction_pct", "mean"),
                fastest_time_sec=("time_sec", "min"),
                mean_time_sec=("time_sec", "mean"),
                comparison_ready_stacks=("comparison_ready", "sum"),
            )
            .reset_index()
        )
    else:
        summary = pd.DataFrame()
    summary.to_csv(table_dir / "objective_dataset_model_scope_summary.csv", index=False)
    summary.to_csv(table_dir / "objective_dataset_model_scope_ratio_summary.csv", index=False)

    if not comparison.empty:
        wide = comparison.pivot_table(
            index=["objective_label", "dataset", "model", "scope", "ratio", "context_rank", "report_stack_id", "singular_method_display"],
            columns="metric",
            values="hybrid_advantage_vs_singular",
            aggfunc="first",
        ).reset_index()
        for col in ["accuracy_delta_pp", "accuracy_pct", "flops_reduction_pct", "time_sec_lower_is_better"]:
            if col in wide.columns:
                wide[f"hybrid_wins_{col}"] = pd.to_numeric(wide[col], errors="coerce") > 0
        win_cols = [c for c in wide.columns if c.startswith("hybrid_wins_")]
        win_summary = (
            wide.groupby(["objective_label", "dataset", "model", "scope", "ratio", "context_rank", "report_stack_id"], dropna=False)
            .agg(singular_methods_compared=("singular_method_display", "nunique"), **{c.replace("hybrid_wins_", "win_rate_"): (c, "mean") for c in win_cols})
            .reset_index()
        )
    else:
        wide = pd.DataFrame()
        win_summary = pd.DataFrame()
    wide.to_csv(table_dir / "hybrid_vs_singular_advantage_wide.csv", index=False)
    win_summary.to_csv(table_dir / "hybrid_vs_singular_win_rates.csv", index=False)

    qc = pd.DataFrame(
        [
            {"check": "top_hybrid_rows", "value": len(top), "status": "ok" if len(top) else "empty"},
            {"check": "layerwise_policy_rows", "value": len(layerwise), "status": "ok" if len(layerwise) else "empty"},
            {"check": "comparison_rows", "value": len(comparison), "status": "ok" if len(comparison) else "empty"},
            {
                "check": "allowed_prune_ratio_filter_exclusions",
                "value": 0 if allowed_ratio_filter_audit.empty else int(allowed_ratio_filter_audit["status"].astype(str).eq("excluded_disallowed_prune_ratio").sum()),
                "status": "ok",
            },
            {
                "check": "latest_family_run_filter_exclusions",
                "value": 0 if latest_family_run_filter_audit.empty else int(latest_family_run_filter_audit["status"].astype(str).eq("excluded_stale_family_run").sum()),
                "status": "ok",
            },
            {"check": "missing_layer_policy_issues", "value": len(policy_audit), "status": "ok" if policy_audit.empty else "review"},
            {
                "check": "local_scope_flops_variability_contexts",
                "value": len(local_flops_variability),
                "status": "ok",
            },
            {"check": "context_match_violations", "value": 0, "status": "ok"},
        ]
    )
    qc.to_csv(table_dir / "qc_summary.csv", index=False)

    index_lines = [
        "# Context-safe reporting artifact index",
        "",
        f"Report output: `{report_dir}`",
        "",
        "## Key tables",
        "- `tables/integrated_top_policy_report.csv`",
        "- `tables/hybrid_layerwise_policy_linked_to_metrics.csv`",
        "- `tables/hybrid_vs_singular_exact_context_long.csv`",
        "- `tables/hybrid_vs_singular_win_rates.csv`",
        "- `tables/baseline_model_scale.csv`",
        "- `tables/objective_dataset_model_scope_summary.csv`",
        "- `tables/local_scope_flops_variability_diagnostic.csv`",
        "- `tables/qc_summary.csv`",
        "",
        "## Key plots",
        "- `plots/layerwise_policies/`",
        "- `plots/hybrid_vs_singular/`",
        "- `plots/absolute_footprints/`",
        "- `plots/pareto/`",
        "- `plots/summary/`",
    ]
    (report_dir / "REPORT_INDEX.md").write_text("\n".join(index_lines), encoding="utf-8")

    manifest = {
        "report_dir": str(report_dir),
        "tables_dir": str(table_dir),
        "figures_dir": str(fig_dir),
        "coverage_plot": str(coverage_plot) if coverage_plot else "",
        "readiness_plot": str(readiness_plot) if readiness_plot else "",
        "counts": {
            "contexts": int(len(contexts)),
            "singular_cache": int(len(singular_cache)),
            "ranked_hybrids": int(len(ranked)),
            "top_hybrids": int(len(top)),
            "layerwise_rows": int(len(layerwise)),
            "comparison_rows": int(len(comparison)),
            "layerwise_plots": int(plot_manifest["layerwise_plot"].astype(bool).sum()) if not plot_manifest.empty else 0,
            "comparison_plots": int(plot_manifest["comparison_plot"].astype(bool).sum()) if not plot_manifest.empty else 0,
            "absolute_footprint_plots": int(plot_manifest.get("absolute_footprint_plot", pd.Series(dtype=str)).astype(bool).sum()) if not plot_manifest.empty else 0,
            "pareto_plots": int(len(pareto_manifest)),
            "region_heatmaps": int(len(region_manifest)),
        },
    }
    with (report_dir / "report_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)
    print(json.dumps(manifest, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs/lfpc_hybrid"))
    parser.add_argument("--registry-dir", type=Path, default=Path("reports/experiment_registry"))
    parser.add_argument("--report-dir", type=Path, default=Path("report_artifacts/context_safe_hybrid_singular_report"))
    parser.add_argument("--v4-report-dir", type=Path, default=Path("report_artifacts/context_safe_hybrid_singular_report_v4_model_metrics"))
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--accuracy-gate-pp", type=float, default=7.0)
    parser.add_argument("--max-plots", type=int, default=None)
    parser.add_argument(
        "--allowed-ratios",
        default=",".join(f"{r:g}" for r in DEFAULT_ALLOWED_PRUNE_RATIOS),
        help="Comma-separated prune ratios to include in context-safe reporting. Use 'all' to disable filtering.",
    )
    parser.add_argument("--refresh-registry", action="store_true")
    parser.add_argument("--skip-plots", action="store_true", help="Build context tables and top-stack selection without rendering figures.")
    return parser.parse_args()


if __name__ == "__main__":
    write_outputs(parse_args())
