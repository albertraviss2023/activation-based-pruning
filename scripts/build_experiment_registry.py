"""Build a project-level LFPC experiment metadata registry.

The registry is a lightweight query layer over notebook outputs. It does not
move or rewrite experiment artifacts; it scans run directories, normalizes the
important context columns, and writes CSV/JSON files that reporting notebooks
can use before analysis. Its main job is context awareness and provenance:
which dataset/model/objective/scope/ratio/settings were produced by which
time-stamped run, and which artifacts belong to that run.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SCHEMA_VERSION = "2026-05-20.1"

RUN_SENTINELS = {
    "run_manifest.json",
    "fixed_hybrid_stack_benchmarks.csv",
    "current_run_singular_method_benchmarks.csv",
    "lfpc_discovered_layer_policy_phase1.csv",
    "method_score_timing.csv",
    "algorithm2_threshold_grid_summary.csv",
}

HYBRID_TABLES = [
    "fixed_hybrid_stack_benchmarks.csv",
    "fixed_hybrid_stack_benchmarks_constraint_passed.csv",
    "fixed_hybrid_stack_benchmarks_all.csv",
    "top_stack_reporting/notebook_top_ranked_hybrid_stacks.csv",
]

SINGULAR_TABLES = [
    "current_run_singular_method_benchmarks.csv",
    "singular_method_benchmarks.csv",
]

CONTEXT_TABLES = [
    "algorithm2_threshold_grid_summary.csv",
    "lfpc_discovered_layer_policy_phase1.csv",
    "method_score_timing.csv",
    "artifact_completeness_audit.csv",
    "top_stack_reporting/notebook_top_stack_vs_same_scope_singular_comparison.csv",
]

OBJECTIVE_LABELS = {
    "flops_accuracy": "FLOPs + Accuracy",
    "time_accuracy": "Time + Accuracy",
    "time_flops": "FLOPs + Time",
    "all_three": "FLOPs + Time + Accuracy",
}

DATASET_ALIASES = {
    "cifar10": "CIFAR-10",
    "cifar-10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "cifar-100": "CIFAR-100",
    "cats_dogs": "Cats-vs-Dogs",
    "cats-vs-dogs": "Cats-vs-Dogs",
    "catdog": "Cats-vs-Dogs",
}

MODEL_ALIASES = {
    "vgg16": "VGG16",
    "resnet18": "ResNet18",
    "mobilenet_v2": "MobileNetV2",
    "mobilenetv2": "MobileNetV2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs/lfpc_hybrid"))
    parser.add_argument("--registry-dir", type=Path, default=Path("reports/experiment_registry"))
    parser.add_argument("--accuracy-gate-pp", type=float, default=7.0)
    parser.add_argument("--top-k", type=int, default=5, help="Optional convenience ranking only; analysis notebooks should own final top-stack logic.")
    parser.add_argument("--max-artifacts-per-run", type=int, default=2500)
    return parser.parse_args()


def safe_slug(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return text or "unknown"


def standardize_dataset(value: Any) -> str:
    text = str(value or "").strip()
    return DATASET_ALIASES.get(text.lower(), text)


def standardize_model(value: Any) -> str:
    text = str(value or "").strip()
    return MODEL_ALIASES.get(text.lower(), text)


def objective_label(value: Any) -> str:
    key = str(value or "").strip().lower()
    return OBJECTIVE_LABELS.get(key, str(value or "unknown"))


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def safe_read_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, nrows=nrows)
    except (pd.errors.EmptyDataError, UnicodeDecodeError, OSError):
        return pd.DataFrame()


def safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return {}


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def metric_series(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    col = first_existing_column(df, candidates)
    if col is None:
        return pd.Series([math.nan] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def metric_series_with_source(df: pd.DataFrame, candidates: Iterable[str]) -> tuple[pd.Series, str]:
    col = first_existing_column(df, candidates)
    if col is None:
        return pd.Series([math.nan] * len(df), index=df.index, dtype="float64"), "not_available"
    return pd.to_numeric(df[col], errors="coerce"), col


def flops_reduction_series_with_source(df: pd.DataFrame) -> tuple[pd.Series, str]:
    """Return FLOPs reduction percent, deriving it from raw FLOPs when needed.

    Some older ResNet notebook exports wrote baseline/pruned/healed FLOPs but
    left the percentage column blank. The registry should preserve the measured
    artifact values instead of treating those contexts as unavailable.
    """
    direct, source = metric_series_with_source(
        df,
        ["healed_flops_reduction_pct", "flops_reduction_pct", "final_flops_reduction_pct", "actual_flops_reduction_pct"],
    )
    baseline_col = first_existing_column(df, ["baseline_flops", "original_flops"])
    final_col = first_existing_column(df, ["healed_flops", "final_flops", "raw_pruned_flops", "pruned_flops"])
    if baseline_col is None or final_col is None:
        return direct, source

    baseline = pd.to_numeric(df[baseline_col], errors="coerce")
    final = pd.to_numeric(df[final_col], errors="coerce")
    derived = ((baseline - final) / baseline.replace(0, math.nan) * 100).where(baseline.gt(0))
    missing = direct.isna() & derived.notna()
    if missing.any():
        filled = direct.copy()
        filled.loc[missing] = derived.loc[missing]
        if direct.notna().any():
            return filled, f"{source}+derived_missing:{baseline_col}-{final_col}"
        return filled, f"derived:{baseline_col}-{final_col}"
    return direct, source


def first_existing_value(row: pd.Series, candidates: Iterable[str], default: Any = "") -> Any:
    for col in candidates:
        if col in row.index and pd.notna(row[col]):
            return row[col]
    return default


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path)


def infer_context_from_path(run_dir: Path, outputs_root: Path) -> dict[str, Any]:
    parts = []
    try:
        parts = list(run_dir.resolve().relative_to(outputs_root.resolve()).parts)
    except ValueError:
        parts = list(run_dir.parts)

    timestamp = next((p for p in reversed(parts) if re.fullmatch(r"20\d{6}_\d{6}", p)), "")
    dataset = standardize_dataset(parts[0]) if len(parts) >= 1 else ""
    model = standardize_model(parts[1]) if len(parts) >= 2 else ""
    objective = next((p for p in parts if p in OBJECTIVE_LABELS), "")
    if not objective:
        objective = "all_three"
    run_id = "_".join(safe_slug(x) for x in ["lfpc", dataset, model, objective, timestamp or run_dir.name] if x)
    return {
        "run_id": run_id,
        "run_dir": relative_path(run_dir),
        "dataset": dataset,
        "model": model,
        "objective": objective,
        "objective_label": objective_label(objective),
        "timestamp": timestamp,
    }


def is_run_dir(path: Path) -> bool:
    return any((path / name).exists() for name in RUN_SENTINELS)


def iter_run_dirs(outputs_root: Path) -> list[Path]:
    if not outputs_root.exists():
        return []
    seen: set[Path] = set()
    run_dirs: list[Path] = []
    for sentinel in RUN_SENTINELS:
        for file_path in outputs_root.rglob(sentinel):
            run_dir = file_path.parent
            if run_dir.name in {"top_stack_reporting", "phase2_phase3_outputs"}:
                run_dir = run_dir.parent
            if run_dir not in seen and is_run_dir(run_dir):
                seen.add(run_dir)
                run_dirs.append(run_dir)
    return sorted(run_dirs, key=lambda p: str(p).lower())


def artifact_role(path: Path) -> str:
    name = path.name.lower()
    parent = path.parent.name.lower()
    if name in {"run_manifest.json", "run_manifest.txt"}:
        return "run_manifest"
    if name.endswith(".pth") or name.endswith(".pt"):
        return "model_checkpoint"
    if name in {"fixed_hybrid_stack_benchmarks.csv", "fixed_hybrid_stack_benchmarks_all.csv"}:
        return "hybrid_benchmark_table"
    if name == "current_run_singular_method_benchmarks.csv":
        return "singular_benchmark_table"
    if name == "lfpc_discovered_layer_policy_phase1.csv":
        return "layer_policy_table"
    if name.startswith("algorithm2_threshold_grid"):
        return "similarity_threshold_table"
    if name == "method_score_timing.csv":
        return "method_scoring_table"
    if name == "artifact_completeness_audit.csv":
        return "quality_audit"
    if parent == "top_stack_reporting":
        return "top_stack_reporting"
    if name.endswith(".png"):
        return "plot"
    if name.endswith(".json"):
        return "json_metadata"
    if name.endswith(".csv"):
        return "csv_table"
    return "other"


def build_artifact_records(run_dir: Path, run: dict[str, Any], max_artifacts: int) -> list[dict[str, Any]]:
    rows = []
    files = [p for p in run_dir.rglob("*") if p.is_file()]
    for path in sorted(files, key=lambda p: str(p).lower())[:max_artifacts]:
        stat = path.stat()
        rows.append({
            **run,
            "artifact_path": relative_path(path),
            "artifact_name": path.name,
            "artifact_role": artifact_role(path),
            "artifact_size_bytes": stat.st_size,
            "artifact_modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        })
    if len(files) > max_artifacts:
        rows.append({
            **run,
            "artifact_path": relative_path(run_dir),
            "artifact_name": "__artifact_scan_truncated__",
            "artifact_role": "scan_note",
            "artifact_size_bytes": 0,
            "artifact_modified_utc": "",
            "note": f"Only first {max_artifacts} files indexed from {len(files)} total files.",
        })
    return rows


def enrich_context_from_df(run: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    ctx = dict(run)
    if not df.empty:
        first = df.iloc[0]
        ctx["dataset"] = standardize_dataset(first_existing_value(first, ["dataset"], ctx.get("dataset", "")))
        ctx["model"] = standardize_model(first_existing_value(first, ["model"], ctx.get("model", "")))
        objective = first_existing_value(first, ["objective", "objective_scenario"], ctx.get("objective", ""))
        if objective:
            ctx["objective"] = str(objective)
            ctx["objective_label"] = str(first_existing_value(first, ["objective_label"], objective_label(objective)))
    return ctx


def normalize_benchmark_rows(df: pd.DataFrame, run: dict[str, Any], source_path: Path, record_type: str) -> list[dict[str, Any]]:
    if df.empty:
        return []
    ctx = enrich_context_from_df(run, df)
    acc, acc_source = metric_series_with_source(df, ["healed_accuracy_pct", "test_accuracy_pct", "final_test_accuracy_pct", "final_test_accuracy", "accuracy_pct"])
    base_acc, base_acc_source = metric_series_with_source(df, ["baseline_test_accuracy_pct", "baseline_validation_accuracy_pct", "baseline_accuracy_pct", "baseline_accuracy"])
    acc_delta, acc_delta_source = metric_series_with_source(df, ["healed_accuracy_delta_pct", "accuracy_delta_pct", "test_accuracy_delta_pct", "val_accuracy_delta_pct"])
    if acc_delta.isna().all() and not acc.isna().all() and not base_acc.isna().all():
        acc_delta = acc - base_acc
        acc_delta_source = f"derived:{acc_source}-{base_acc_source}"
    flops, flops_source = flops_reduction_series_with_source(df)
    params = metric_series(df, ["healed_params_reduction_pct", "params_reduction_pct", "final_params_reduction_pct", "actual_params_reduction_pct"])
    runtime = metric_series(df, ["end_to_end_pruning_time_sec", "fixed_stack_end_to_end_pruning_time_sec", "total_time_sec", "pruning_time_sec"])

    rows: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        scope = str(first_existing_value(row, ["scope"], ""))
        ratio = safe_float(first_existing_value(row, ["ratio", "prune_ratio"], math.nan))
        stack_or_method = first_existing_value(row, ["stack_id", "method_or_stack", "method", "hybrid_stack_id"], "")
        checkpoint = first_existing_value(row, ["pruned_model_checkpoint_path", "checkpoint_path", "model_checkpoint_path"], "")
        rows.append({
            **ctx,
            "record_type": record_type,
            "source_table": relative_path(source_path),
            "source_row": int(idx),
            "scope": scope,
            "ratio": ratio,
            "variance_threshold": safe_float(first_existing_value(row, ["variance_threshold"], math.nan)),
            "spearman_threshold": safe_float(first_existing_value(row, ["spearman_threshold"], math.nan)),
            "jaccard_threshold": safe_float(first_existing_value(row, ["jaccard_threshold"], math.nan)),
            "strategy_type": str(first_existing_value(row, ["strategy_type"], "hybrid_fixed_stack" if record_type == "hybrid" else "singular_fixed_method")),
            "method": str(first_existing_value(row, ["method", "compared_singular_method"], "")),
            "method_or_stack": str(stack_or_method),
            "stack_id": str(first_existing_value(row, ["stack_id", "hybrid_stack_id"], stack_or_method)),
            "selected_methods": str(first_existing_value(row, ["selected_methods", "selected_methods_in_layer_order", "stack_composition"], "")),
            "accuracy_pct": safe_float(acc.loc[idx]),
            "baseline_accuracy_pct": safe_float(base_acc.loc[idx]),
            "accuracy_delta_pp": safe_float(acc_delta.loc[idx]),
            "accuracy_source_column": acc_source,
            "baseline_accuracy_source_column": base_acc_source,
            "accuracy_delta_source_column": acc_delta_source,
            "flops_reduction_pct": safe_float(flops.loc[idx]),
            "flops_reduction_source_column": flops_source,
            "params_reduction_pct": safe_float(params.loc[idx]),
            "time_sec": safe_float(runtime.loc[idx]),
            "checkpoint_path": str(checkpoint),
            "has_checkpoint_path": bool(str(checkpoint).strip() and str(checkpoint).lower() not in {"nan", "none"}),
        })
    return rows


def read_first_existing_table(run_dir: Path, names: list[str]) -> tuple[pd.DataFrame, Path | None]:
    for name in names:
        path = run_dir / name
        df = safe_read_csv(path)
        if not df.empty:
            return df, path
    return pd.DataFrame(), None


def build_run_record(run_dir: Path, outputs_root: Path) -> dict[str, Any]:
    run = infer_context_from_path(run_dir, outputs_root)
    manifest = safe_read_json(run_dir / "run_manifest.json")
    if manifest:
        run["run_id"] = str(manifest.get("run_id") or run["run_id"])
        run["dataset"] = standardize_dataset(manifest.get("dataset_key") or manifest.get("dataset") or run["dataset"])
        run["model"] = standardize_model(manifest.get("model_target") or manifest.get("model") or run["model"])
        run["objective"] = str(manifest.get("objective_scenario") or manifest.get("objective") or run["objective"])
        run["objective_label"] = str(manifest.get("objective_label") or objective_label(run["objective"]))
        run["timestamp"] = str(manifest.get("run_stamp") or manifest.get("timestamp") or run["timestamp"])
        run["manifest_path"] = relative_path(run_dir / "run_manifest.json")
        run["manifest_schema_version"] = str(manifest.get("schema_version", ""))
        run["notebook_path"] = str(manifest.get("notebook_path", ""))
    else:
        run["manifest_path"] = ""
        run["manifest_schema_version"] = ""
        run["notebook_path"] = ""
    candidate_tables = HYBRID_TABLES + SINGULAR_TABLES + CONTEXT_TABLES
    for table_name in candidate_tables:
        df = safe_read_csv(run_dir / table_name, nrows=1)
        if not df.empty:
            # Manifest values are authoritative for run-level identity. CSVs fill
            # gaps for legacy runs that predate manifests.
            if not manifest:
                run = enrich_context_from_df(run, df)
            break
    run["run_modified_utc"] = datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc).isoformat()
    return run


def summarize_run(run: dict[str, Any], run_dir: Path, contexts: list[dict[str, Any]]) -> dict[str, Any]:
    relevant = [r for r in contexts if r["run_id"] == run["run_id"]]
    hybrid = [r for r in relevant if r["record_type"] == "hybrid"]
    singular = [r for r in relevant if r["record_type"] == "singular"]
    audit = safe_read_csv(run_dir / "artifact_completeness_audit.csv")
    blocking_statuses = {"schema_missing", "metric_column_missing", "metric_values_missing", "scope_violation"}
    blocking = 0
    if not audit.empty and "status" in audit.columns:
        blocking = int(audit["status"].astype(str).isin(blocking_statuses).sum())
    return {
        **run,
        "hybrid_context_rows": len(hybrid),
        "singular_context_rows": len(singular),
        "scopes": "|".join(sorted({str(r.get("scope", "")) for r in relevant if str(r.get("scope", ""))})),
        "ratios": "|".join(sorted({f"{safe_float(r.get('ratio')):g}" for r in relevant if math.isfinite(safe_float(r.get("ratio")))})),
        "max_accuracy_delta_pp": max([safe_float(r.get("accuracy_delta_pp")) for r in hybrid] or [math.nan]),
        "max_accuracy_pct": max([safe_float(r.get("accuracy_pct")) for r in hybrid] or [math.nan]),
        "max_flops_reduction_pct": max([safe_float(r.get("flops_reduction_pct")) for r in hybrid] or [math.nan]),
        "min_time_sec": min([safe_float(r.get("time_sec")) for r in hybrid if math.isfinite(safe_float(r.get("time_sec")))] or [math.nan]),
        "audit_blocking_issue_count": blocking,
    }


def objective_score(group: pd.DataFrame, accuracy_gate_pp: float) -> pd.Series:
    def norm_high(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        lo, hi = s.min(), s.max()
        if not math.isfinite(safe_float(lo)) or not math.isfinite(safe_float(hi)) or abs(float(hi) - float(lo)) < 1e-12:
            return pd.Series(0.5, index=s.index)
        return (s - lo) / (hi - lo)

    def norm_low(s: pd.Series) -> pd.Series:
        return 1.0 - norm_high(s)

    acc_delta = pd.to_numeric(group["accuracy_delta_pp"], errors="coerce")
    acc_score = norm_high(acc_delta)
    flops_score = norm_high(group["flops_reduction_pct"])
    time_score = norm_low(group["time_sec"])
    objective = str(group["objective"].iloc[0]).lower() if "objective" in group.columns and len(group) else ""
    
    # Revised weights matching build_context_safe_report.py
    if "flops_accuracy" in objective:
        score = 0.35 * acc_score + 0.65 * flops_score
    elif "time_accuracy" in objective:
        score = 0.35 * acc_score + 0.65 * time_score
    elif "time_flops" in objective:
        score = 0.10 * acc_score + 0.45 * flops_score + 0.45 * time_score
    else:
        # all_three or unknown
        score = 0.20 * acc_score + 0.40 * flops_score + 0.40 * time_score
        
    gate = acc_delta >= -float(accuracy_gate_pp)
    return score.where(gate, score - 1.0)


def rank_best_hybrids(contexts_df: pd.DataFrame, accuracy_gate_pp: float, top_k: int) -> pd.DataFrame:
    if contexts_df.empty:
        return pd.DataFrame()
    hybrids = contexts_df[contexts_df["record_type"] == "hybrid"].copy()
    if hybrids.empty:
        return pd.DataFrame()
    for col in ["ratio", "accuracy_delta_pp", "accuracy_pct", "flops_reduction_pct", "time_sec"]:
        hybrids[col] = pd.to_numeric(hybrids[col], errors="coerce")
    hybrids["accuracy_gate_passed"] = hybrids["accuracy_delta_pp"] >= -float(accuracy_gate_pp)
    group_cols = ["objective", "dataset", "model", "scope", "ratio"]
    ranked_parts = []
    for _, group in hybrids.groupby(group_cols, dropna=False):
        group = group.copy()
        group["registry_objective_score"] = objective_score(group, accuracy_gate_pp)
        group = group.sort_values(
            ["accuracy_gate_passed", "registry_objective_score", "accuracy_delta_pp", "accuracy_pct", "flops_reduction_pct", "time_sec"],
            ascending=[False, False, False, False, False, True],
            na_position="last",
        )
        group["registry_rank_within_context"] = range(1, len(group) + 1)
        ranked_parts.append(group.head(top_k))
    return pd.concat(ranked_parts, ignore_index=True) if ranked_parts else pd.DataFrame()


def build_context_summary(contexts_df: pd.DataFrame) -> pd.DataFrame:
    if contexts_df.empty:
        return pd.DataFrame()
    work = contexts_df.copy()
    for col in ["ratio", "accuracy_delta_pp", "accuracy_pct", "flops_reduction_pct", "time_sec"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    rows: list[dict[str, Any]] = []
    group_cols = ["objective", "objective_label", "dataset", "model", "scope", "ratio"]
    for keys, group in work.groupby(group_cols, dropna=False):
        rec = {col: val for col, val in zip(group_cols, keys)}
        hybrid = group[group["record_type"] == "hybrid"]
        singular = group[group["record_type"] == "singular"]
        rec.update({
            "hybrid_rows": int(len(hybrid)),
            "singular_rows": int(len(singular)),
            "hybrid_runs": int(hybrid["run_id"].nunique()) if not hybrid.empty else 0,
            "singular_runs": int(singular["run_id"].nunique()) if not singular.empty else 0,
            "hybrid_checkpoint_coverage": float(hybrid["has_checkpoint_path"].fillna(False).mean()) if not hybrid.empty else math.nan,
            "singular_checkpoint_coverage": float(singular["has_checkpoint_path"].fillna(False).mean()) if not singular.empty else math.nan,
            "best_hybrid_accuracy_delta_pp": safe_float(hybrid["accuracy_delta_pp"].max()) if not hybrid.empty else math.nan,
            "best_hybrid_accuracy_pct": safe_float(hybrid["accuracy_pct"].max()) if not hybrid.empty else math.nan,
            "best_hybrid_flops_reduction_pct": safe_float(hybrid["flops_reduction_pct"].max()) if not hybrid.empty else math.nan,
            "best_hybrid_time_sec": safe_float(hybrid["time_sec"].min()) if not hybrid.empty else math.nan,
            "best_singular_accuracy_delta_pp": safe_float(singular["accuracy_delta_pp"].max()) if not singular.empty else math.nan,
            "best_singular_accuracy_pct": safe_float(singular["accuracy_pct"].max()) if not singular.empty else math.nan,
            "best_singular_flops_reduction_pct": safe_float(singular["flops_reduction_pct"].max()) if not singular.empty else math.nan,
            "best_singular_time_sec": safe_float(singular["time_sec"].min()) if not singular.empty else math.nan,
        })
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(group_cols, na_position="last")


def build_context_run_index(contexts_df: pd.DataFrame) -> pd.DataFrame:
    """One row per run-specific context.

    This is the registry's most important analysis-facing file: it tells a
    reporting notebook which exact run produced rows for a context, preserving
    timestamp instead of collapsing across runs.
    """
    if contexts_df.empty:
        return pd.DataFrame()
    work = contexts_df.copy()
    for col in ["ratio", "variance_threshold", "spearman_threshold", "jaccard_threshold"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    group_cols = [
        "run_id", "run_dir", "timestamp", "run_modified_utc",
        "objective", "objective_label", "dataset", "model", "scope", "ratio",
        "variance_threshold", "spearman_threshold", "jaccard_threshold",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in work.groupby(group_cols, dropna=False):
        rec = {col: val for col, val in zip(group_cols, keys)}
        hybrid = group[group["record_type"] == "hybrid"]
        singular = group[group["record_type"] == "singular"]
        rec.update({
            "hybrid_rows": int(len(hybrid)),
            "singular_rows": int(len(singular)),
            "artifact_source_tables": "|".join(sorted(set(group["source_table"].dropna().astype(str)))),
            "hybrid_checkpoint_rows": int(hybrid["has_checkpoint_path"].fillna(False).astype(bool).sum()) if not hybrid.empty else 0,
            "singular_checkpoint_rows": int(singular["has_checkpoint_path"].fillna(False).astype(bool).sum()) if not singular.empty else 0,
            "context_key": context_key(rec),
            "run_context_key": run_context_key(rec),
        })
        rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["dataset", "model", "objective", "scope", "ratio", "timestamp", "run_modified_utc"], na_position="last")


def context_key(row: dict[str, Any] | pd.Series) -> str:
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


def run_context_key(row: dict[str, Any] | pd.Series) -> str:
    return "__".join([safe_slug(row.get("run_id", "")), context_key(row)])


def build_latest_context_runs(context_run_df: pd.DataFrame) -> pd.DataFrame:
    """Latest run per exact context key, based on timestamp then modified time."""
    if context_run_df.empty:
        return pd.DataFrame()
    work = context_run_df.copy()
    work["_timestamp_sort"] = work["timestamp"].fillna("").astype(str)
    work["_modified_sort"] = work["run_modified_utc"].fillna("").astype(str)
    work = work.sort_values(["context_key", "_timestamp_sort", "_modified_sort"], ascending=[True, False, False])
    latest = work.drop_duplicates("context_key", keep="first").drop(columns=["_timestamp_sort", "_modified_sort"], errors="ignore")
    return latest.sort_values(["dataset", "model", "objective", "scope", "ratio"], na_position="last")


def singular_cache_key(row: dict[str, Any] | pd.Series) -> str:
    parts = [
        row.get("dataset", ""),
        row.get("model", ""),
        row.get("scope", ""),
        f"r{safe_float(row.get('ratio')):g}" if math.isfinite(safe_float(row.get("ratio"))) else "rNA",
        row.get("method", "") or row.get("method_or_stack", ""),
    ]
    return "__".join(safe_slug(p) for p in parts)


def build_singular_cache_index(contexts_df: pd.DataFrame) -> pd.DataFrame:
    """Latest reusable singular benchmark/checkpoint per exact singular context.

    Singular prunes are expensive and intentionally reused across objective
    notebooks. This index makes that reuse explicit: later notebooks can load
    singular rows/checkpoints by dataset, model, scope, ratio, and method while
    retaining the source run that originally produced them.
    """
    if contexts_df.empty:
        return pd.DataFrame()
    singular = contexts_df[contexts_df["record_type"] == "singular"].copy()
    if singular.empty:
        return pd.DataFrame()
    singular["cache_key"] = singular.apply(singular_cache_key, axis=1)
    singular["_has_metrics"] = (
        pd.to_numeric(singular["accuracy_delta_pp"], errors="coerce").notna()
        & pd.to_numeric(singular["flops_reduction_pct"], errors="coerce").notna()
        & pd.to_numeric(singular["time_sec"], errors="coerce").notna()
    )
    singular["_has_checkpoint"] = singular["has_checkpoint_path"].fillna(False).astype(bool)
    singular["_timestamp_sort"] = singular["timestamp"].fillna("").astype(str)
    singular["_modified_sort"] = singular["run_modified_utc"].fillna("").astype(str)
    singular = singular.sort_values(
        ["cache_key", "_has_checkpoint", "_has_metrics", "_timestamp_sort", "_modified_sort"],
        ascending=[True, False, False, False, False],
    )
    latest = singular.drop_duplicates("cache_key", keep="first").copy()
    latest["cache_source_run_id"] = latest["run_id"]
    latest["cache_source_run_stamp"] = latest["timestamp"]
    latest = latest.drop(columns=["_has_metrics", "_has_checkpoint", "_timestamp_sort", "_modified_sort"], errors="ignore")
    return latest.sort_values(["dataset", "model", "scope", "ratio", "method"], na_position="last")


def build_quality_audit(contexts_df: pd.DataFrame, artifacts_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if contexts_df.empty:
        rows.append({"severity": "error", "issue_type": "empty_contexts", "issue": "No benchmark context rows were indexed."})
        return pd.DataFrame(rows)

    metric_cols = ["accuracy_delta_pp", "accuracy_pct", "flops_reduction_pct", "time_sec"]
    for record_type, group in contexts_df.groupby("record_type", dropna=False):
        for col in metric_cols:
            missing = int(pd.to_numeric(group[col], errors="coerce").isna().sum()) if col in group.columns else len(group)
            if missing:
                rows.append({
                    "severity": "warning",
                    "issue_type": "missing_metric",
                    "record_type": record_type,
                    "metric": col,
                    "issue": f"{missing}/{len(group)} {record_type} rows are missing {col}.",
                })
        if "has_checkpoint_path" in group.columns:
            missing_ckpt = int((~group["has_checkpoint_path"].fillna(False).astype(bool)).sum())
            if missing_ckpt:
                rows.append({
                    "severity": "warning",
                    "issue_type": "missing_checkpoint_path",
                    "record_type": record_type,
                    "issue": f"{missing_ckpt}/{len(group)} {record_type} rows have no checkpoint path in the source table.",
                })

    key_cols = ["dataset", "model", "scope", "ratio"]
    missing_key = contexts_df[key_cols].isna().any(axis=1) | contexts_df[key_cols].astype(str).isin(["", "nan"]).any(axis=1)
    if missing_key.any():
        rows.append({
            "severity": "warning",
            "issue_type": "missing_context_key",
            "issue": f"{int(missing_key.sum())}/{len(contexts_df)} rows are missing dataset/model/scope/ratio context.",
        })

    if not artifacts_df.empty and "artifact_role" in artifacts_df.columns:
        checkpoint_count = int((artifacts_df["artifact_role"] == "model_checkpoint").sum())
        rows.append({
            "severity": "info",
            "issue_type": "checkpoint_artifact_count",
            "issue": f"{checkpoint_count} checkpoint artifact files were indexed from run directories.",
        })
    return pd.DataFrame(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    outputs_root = args.outputs_root
    registry_dir = args.registry_dir
    registry_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = iter_run_dirs(outputs_root)
    all_contexts: list[dict[str, Any]] = []
    all_artifacts: list[dict[str, Any]] = []
    runs_base: list[tuple[dict[str, Any], Path]] = []

    for run_dir in run_dirs:
        run = build_run_record(run_dir, outputs_root)
        runs_base.append((run, run_dir))
        hybrid_df, hybrid_path = read_first_existing_table(run_dir, HYBRID_TABLES)
        if hybrid_path is not None:
            all_contexts.extend(normalize_benchmark_rows(hybrid_df, run, hybrid_path, "hybrid"))
        singular_df, singular_path = read_first_existing_table(run_dir, SINGULAR_TABLES)
        if singular_path is not None:
            all_contexts.extend(normalize_benchmark_rows(singular_df, run, singular_path, "singular"))
        all_artifacts.extend(build_artifact_records(run_dir, run, args.max_artifacts_per_run))

    run_rows = [summarize_run(run, run_dir, all_contexts) for run, run_dir in runs_base]
    runs_df = pd.DataFrame(run_rows)
    artifacts_df = pd.DataFrame(all_artifacts)
    contexts_df = pd.DataFrame(all_contexts)
    best_df = rank_best_hybrids(contexts_df, args.accuracy_gate_pp, args.top_k)
    context_summary_df = build_context_summary(contexts_df)
    context_run_df = build_context_run_index(contexts_df)
    latest_context_runs_df = build_latest_context_runs(context_run_df)
    singular_cache_df = build_singular_cache_index(contexts_df)
    quality_audit_df = build_quality_audit(contexts_df, artifacts_df)

    runs_df.to_csv(registry_dir / "runs.csv", index=False)
    artifacts_df.to_csv(registry_dir / "artifacts.csv", index=False)
    contexts_df.to_csv(registry_dir / "contexts.csv", index=False)
    context_run_df.to_csv(registry_dir / "context_run_index.csv", index=False)
    latest_context_runs_df.to_csv(registry_dir / "latest_context_runs.csv", index=False)
    singular_cache_df.to_csv(registry_dir / "singular_cache_index.csv", index=False)
    best_df.to_csv(registry_dir / "best_hybrid_by_context.csv", index=False)
    context_summary_df.to_csv(registry_dir / "context_summary.csv", index=False)
    quality_audit_df.to_csv(registry_dir / "registry_quality_audit.csv", index=False)

    write_jsonl(registry_dir / "runs.jsonl", run_rows)
    write_jsonl(registry_dir / "artifacts.jsonl", all_artifacts)
    write_jsonl(registry_dir / "contexts.jsonl", all_contexts)
    write_jsonl(registry_dir / "context_run_index.jsonl", context_run_df.to_dict(orient="records") if not context_run_df.empty else [])
    write_jsonl(registry_dir / "latest_context_runs.jsonl", latest_context_runs_df.to_dict(orient="records") if not latest_context_runs_df.empty else [])
    write_jsonl(registry_dir / "singular_cache_index.jsonl", singular_cache_df.to_dict(orient="records") if not singular_cache_df.empty else [])
    write_jsonl(registry_dir / "best_hybrid_by_context.jsonl", best_df.to_dict(orient="records") if not best_df.empty else [])
    write_jsonl(registry_dir / "context_summary.jsonl", context_summary_df.to_dict(orient="records") if not context_summary_df.empty else [])

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs_root": relative_path(outputs_root),
        "registry_dir": relative_path(registry_dir),
        "accuracy_gate_pp": args.accuracy_gate_pp,
        "top_k_per_context": args.top_k,
        "counts": {
            "run_dirs": len(run_dirs),
            "runs": int(len(runs_df)),
            "artifacts": int(len(artifacts_df)),
            "contexts": int(len(contexts_df)),
            "context_run_rows": int(len(context_run_df)),
            "latest_context_rows": int(len(latest_context_runs_df)),
            "singular_cache_rows": int(len(singular_cache_df)),
            "context_summary_rows": int(len(context_summary_df)),
            "best_hybrid_rows": int(len(best_df)),
            "quality_audit_rows": int(len(quality_audit_df)),
        },
        "primary_files": {
            "runs": "runs.csv",
            "artifacts": "artifacts.csv",
            "contexts": "contexts.csv",
            "context_run_index": "context_run_index.csv",
            "latest_context_runs": "latest_context_runs.csv",
            "singular_cache_index": "singular_cache_index.csv",
            "context_summary": "context_summary.csv",
            "best_hybrid_by_context": "best_hybrid_by_context.csv (convenience only; final ranking belongs in analysis notebooks)",
            "registry_quality_audit": "registry_quality_audit.csv",
        },
    }
    with (registry_dir / "registry_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
