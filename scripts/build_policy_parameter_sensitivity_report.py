"""Build context-aware sensitivity analysis for LFPC threshold parameters.

This report is intentionally separate from the hybrid-vs-singular thesis plots.
It answers a different question: when Algorithm-2 similarity settings change,
do we keep discovering the same layerwise policy, and how much do the final
metrics move?
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = PROJECT_ROOT / "reports" / "experiment_registry" / "contexts.csv"
DEFAULT_V4_METRICS = (
    PROJECT_ROOT
    / "report_artifacts"
    / "context_safe_hybrid_singular_report_v4_model_metrics"
    / "tables"
    / "v4_checkpoint_direct_model_metrics.csv"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "report_artifacts" / "policy_parameter_sensitivity_report"

CONTEXT_COLS = ["objective", "objective_label", "dataset", "model", "scope", "ratio"]
PARAM_COLS = ["variance_threshold", "spearman_threshold", "jaccard_threshold"]
EXACT_COLS = CONTEXT_COLS + PARAM_COLS


METHOD_DISPLAY = {
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
    "custom_senpips": "SeNPIS",
    "custom_gfs": "GFS",
}


def method_display(value: object) -> str:
    text = str(value)
    return METHOD_DISPLAY.get(text, text)


def safe_slug(value: object, max_len: int = 140) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return (text or "item")[:max_len]


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def to_num(series: pd.Series | object) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def norm_high(series: pd.Series) -> pd.Series:
    s = to_num(series)
    lo, hi = s.min(skipna=True), s.max(skipna=True)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(float(hi) - float(lo)) < 1e-12:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def norm_low(series: pd.Series) -> pd.Series:
    return 1.0 - norm_high(series)


THESIS_COLORS = {
    "FLOPs + Accuracy": "#0E9F6E",
    "Time + Accuracy": "#E76F51",
    "FLOPs + Time + Accuracy": "#2563EB",
    "Time + FLOPs": "#7C3AED",
}


def apply_thesis_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#334155",
            "axes.linewidth": 0.8,
            "axes.titleweight": "semibold",
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.titlesize": 15,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
        }
    )


def wrapped_label(text: object, width: int = 28) -> str:
    return "\n".join(wrap(str(text), width=width, break_long_words=False, replace_whitespace=False))


def row_value(row: pd.Series | object, name: str):
    if isinstance(row, pd.Series):
        return row.get(name)
    return getattr(row, name)


def context_label(row: pd.Series | object, width: int = 35) -> str:
    text = f"{row_value(row, 'objective_label')} | {row_value(row, 'dataset')} | {row_value(row, 'model')} | {row_value(row, 'scope')} | r={float(row_value(row, 'ratio')):g}"
    return wrapped_label(text, width)


def prettify_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#CBD5E1", alpha=0.35, linewidth=0.7)
    ax.set_axisbelow(True)


def entropy_from_counts(counts: pd.Series) -> tuple[float, float, float]:
    total = float(counts.sum())
    if total <= 0:
        return np.nan, np.nan, np.nan
    p = counts.astype(float) / total
    entropy = float(-(p * np.log(p + 1e-12)).sum())
    normalized = float(entropy / np.log(len(p))) if len(p) > 1 else 0.0
    effective = float(np.exp(entropy))
    return entropy, normalized, effective


def parse_methods(value: object) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value]
    if not isinstance(value, str):
        return []
    try:
        import ast

        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple, set)):
            return [str(x) for x in parsed]
    except Exception:
        pass
    return [x.strip().strip("'\"[](){}") for x in re.split(r"[,+|]", value) if x.strip()]


def objective_score(frame: pd.DataFrame) -> pd.Series:
    work = frame.copy()
    acc = norm_high(work["accuracy_delta_pp"])
    flops = norm_high(work["flops_reduction_pct"])
    time = norm_low(work["time_sec"])
    obj = work["objective"].astype(str)
    score = pd.Series(index=work.index, dtype=float)
    score.loc[obj.eq("flops_accuracy")] = 0.50 * acc + 0.50 * flops
    score.loc[obj.eq("time_accuracy")] = 0.55 * acc + 0.45 * time
    score.loc[obj.eq("all_three")] = 0.40 * acc + 0.35 * flops + 0.25 * time
    score = score.fillna(0.40 * acc + 0.35 * flops + 0.25 * time)
    return score


def latest_hybrid_contexts(registry_path: Path, v4_metrics_path: Path | None) -> pd.DataFrame:
    registry = read_csv_safe(registry_path)
    if registry.empty:
        return registry
    hybrid = registry[registry.get("record_type", "").astype(str).eq("hybrid")].copy()
    hybrid = hybrid.dropna(subset=PARAM_COLS, how="any")
    for col in ["ratio"] + PARAM_COLS + ["accuracy_pct", "baseline_accuracy_pct", "accuracy_delta_pp", "flops_reduction_pct", "params_reduction_pct", "time_sec"]:
        if col in hybrid.columns:
            hybrid[col] = pd.to_numeric(hybrid[col], errors="coerce")
    hybrid["_modified_sort"] = pd.to_datetime(hybrid.get("run_modified_utc"), errors="coerce")
    hybrid["_timestamp_sort"] = hybrid.get("timestamp", "").astype(str)
    hybrid = hybrid.sort_values(["_modified_sort", "_timestamp_sort"], na_position="first")
    hybrid = hybrid.drop_duplicates(EXACT_COLS, keep="last").copy()

    if v4_metrics_path and v4_metrics_path.exists():
        v4 = read_csv_safe(v4_metrics_path)
        if not v4.empty:
            v4 = v4[v4.get("record_type", "").astype(str).eq("hybrid")].copy()
            v4 = v4.dropna(subset=PARAM_COLS, how="any")
            for col in ["ratio"] + PARAM_COLS + ["direct_flops_reduction_pct", "direct_params_reduction_pct"]:
                if col in v4.columns:
                    v4[col] = pd.to_numeric(v4[col], errors="coerce")
            keep = EXACT_COLS + ["stack_id", "direct_flops_reduction_pct", "direct_params_reduction_pct", "metric_status"]
            keep = [c for c in keep if c in v4.columns]
            v4 = v4[keep].drop_duplicates(EXACT_COLS + ["stack_id"], keep="last")
            hybrid = hybrid.merge(v4, on=EXACT_COLS + ["stack_id"], how="left")
            missing_flops = hybrid["flops_reduction_pct"].isna() & hybrid.get("direct_flops_reduction_pct", pd.Series(index=hybrid.index)).notna()
            hybrid.loc[missing_flops, "flops_reduction_pct"] = hybrid.loc[missing_flops, "direct_flops_reduction_pct"]
            hybrid.loc[missing_flops, "flops_reduction_source_column"] = "checkpoint_direct_v4"
            missing_params = hybrid["params_reduction_pct"].isna() & hybrid.get("direct_params_reduction_pct", pd.Series(index=hybrid.index)).notna()
            hybrid.loc[missing_params, "params_reduction_pct"] = hybrid.loc[missing_params, "direct_params_reduction_pct"]
    hybrid["analysis_score"] = objective_score(hybrid)
    return hybrid


def load_policy_rows(hybrid: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    def norm_key(value: object) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).lower())

    for run_dir, run_group in hybrid.groupby("run_dir", dropna=False):
        if not isinstance(run_dir, str) or not run_dir:
            continue
        policy_path = project_root / run_dir / "lfpc_discovered_layer_policy_phase1.csv"
        policy = read_csv_safe(policy_path)
        if policy.empty:
            continue
        for col in ["ratio"] + PARAM_COLS:
            if col in policy.columns:
                policy[col] = pd.to_numeric(policy[col], errors="coerce")
        policy["_dataset_key"] = policy["dataset"].map(norm_key)
        policy["_model_key"] = policy["model"].map(norm_key)
        keys = run_group[["run_id", "run_dir", "objective", "objective_label", "dataset", "model", "scope", "ratio"] + PARAM_COLS + ["stack_id"]].drop_duplicates()
        keys["_dataset_key"] = keys["dataset"].map(norm_key)
        keys["_model_key"] = keys["model"].map(norm_key)
        merged = policy.merge(
            keys,
            on=["_dataset_key", "_model_key", "scope", "ratio"] + PARAM_COLS + ["stack_id"],
            how="inner",
            suffixes=("", "_context"),
        )
        if not merged.empty:
            for col in ["dataset", "model"]:
                ctx_col = f"{col}_context"
                if ctx_col in merged.columns:
                    merged[col] = merged[ctx_col]
            rows.append(merged)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    if "selected_method_display" not in out.columns:
        out["selected_method_display"] = out["selected_method"].map(method_display)
    return out


def build_policy_signatures(hybrid: pd.DataFrame, policy_rows: pd.DataFrame) -> pd.DataFrame:
    work = hybrid.copy()
    if not policy_rows.empty and {"stack_id", "layer", "selected_method"}.issubset(policy_rows.columns):
        p = policy_rows.copy()
        p["layer"] = p["layer"].astype(str)
        p["selected_method"] = p["selected_method"].astype(str)
        p = p.sort_values(EXACT_COLS + ["stack_id", "layer"], kind="mergesort")
        sig = (
            p.groupby(EXACT_COLS + ["stack_id"], dropna=False)
            .agg(
                policy_signature=("selected_method", lambda s: "|".join(s.astype(str))),
                layer_signature=("layer", lambda s: "|".join(s.astype(str))),
                num_layers=("layer", "nunique"),
                methods_used=("selected_method_display", lambda s: " + ".join(pd.Series(s).dropna().astype(str).drop_duplicates())),
            )
            .reset_index()
        )
        work = work.merge(sig, on=EXACT_COLS + ["stack_id"], how="left")
    if "policy_signature" not in work.columns:
        work["policy_signature"] = np.nan
    if "methods_used" not in work.columns:
        work["methods_used"] = np.nan
    if "num_layers" not in work.columns:
        work["num_layers"] = np.nan
    work["policy_signature"] = work["policy_signature"].fillna(work["selected_methods"].map(lambda x: "|".join(parse_methods(x))))
    work["methods_used"] = work["methods_used"].fillna(work["selected_methods"].map(lambda x: " + ".join(method_display(m) for m in parse_methods(x))))
    work["num_layers"] = pd.to_numeric(work.get("num_layers", np.nan), errors="coerce")
    return work


def pairwise_policy_agreement(signatures: Iterable[str]) -> float:
    sigs = [str(s).split("|") for s in signatures if isinstance(s, str) and s]
    if len(sigs) < 2:
        return 1.0 if len(sigs) == 1 else np.nan
    agreements = []
    for i in range(len(sigs)):
        for j in range(i + 1, len(sigs)):
            n = min(len(sigs[i]), len(sigs[j]))
            if n <= 0:
                continue
            agreements.append(sum(a == b for a, b in zip(sigs[i][:n], sigs[j][:n])) / n)
    return float(np.mean(agreements)) if agreements else np.nan


def stability_tables(hybrid: pd.DataFrame, policy_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    for keys, grp in hybrid.groupby(CONTEXT_COLS, dropna=False):
        signatures = grp["policy_signature"].fillna("").astype(str)
        counts = signatures.value_counts(dropna=False)
        modal_sig = counts.index[0] if not counts.empty else ""
        modal_count = int(counts.iloc[0]) if not counts.empty else 0
        n = int(len(grp))
        acc = pd.to_numeric(grp["accuracy_delta_pp"], errors="coerce")
        flops = pd.to_numeric(grp["flops_reduction_pct"], errors="coerce")
        params = pd.to_numeric(grp["params_reduction_pct"], errors="coerce")
        time = pd.to_numeric(grp["time_sec"], errors="coerce")
        score = pd.to_numeric(grp["analysis_score"], errors="coerce")
        best = grp.sort_values(["analysis_score", "accuracy_delta_pp", "flops_reduction_pct", "time_sec"], ascending=[False, False, False, True], na_position="last").iloc[0]
        rec = dict(zip(CONTEXT_COLS, keys))
        rec.update(
            {
                "num_threshold_settings": n,
                "num_unique_policy_signatures": int(counts.size),
                "modal_policy_count": modal_count,
                "modal_policy_share": modal_count / n if n else np.nan,
                "mean_pairwise_layer_agreement": pairwise_policy_agreement(signatures),
                "accuracy_delta_mean": float(acc.mean()) if acc.notna().any() else np.nan,
                "accuracy_delta_std": float(acc.std(ddof=0)) if acc.notna().any() else np.nan,
                "accuracy_delta_min": float(acc.min()) if acc.notna().any() else np.nan,
                "accuracy_delta_max": float(acc.max()) if acc.notna().any() else np.nan,
                "accuracy_delta_range": float(acc.max() - acc.min()) if acc.notna().any() else np.nan,
                "flops_reduction_mean": float(flops.mean()) if flops.notna().any() else np.nan,
                "flops_reduction_std": float(flops.std(ddof=0)) if flops.notna().any() else np.nan,
                "flops_reduction_range": float(flops.max() - flops.min()) if flops.notna().any() else np.nan,
                "params_reduction_mean": float(params.mean()) if params.notna().any() else np.nan,
                "params_reduction_range": float(params.max() - params.min()) if params.notna().any() else np.nan,
                "time_sec_mean": float(time.mean()) if time.notna().any() else np.nan,
                "time_sec_std": float(time.std(ddof=0)) if time.notna().any() else np.nan,
                "time_sec_range": float(time.max() - time.min()) if time.notna().any() else np.nan,
                "analysis_score_mean": float(score.mean()) if score.notna().any() else np.nan,
                "analysis_score_range": float(score.max() - score.min()) if score.notna().any() else np.nan,
                "best_stack_id": best.get("stack_id"),
                "best_accuracy_delta_pp": best.get("accuracy_delta_pp"),
                "best_flops_reduction_pct": best.get("flops_reduction_pct"),
                "best_params_reduction_pct": best.get("params_reduction_pct"),
                "best_time_sec": best.get("time_sec"),
                "best_methods_used": best.get("methods_used"),
                "stability_label": "stable" if modal_count / n >= 0.8 else "moderate" if modal_count / n >= 0.6 else "unstable",
            }
        )
        summary_rows.append(rec)
    summary = pd.DataFrame(summary_rows)

    layer_rows = []
    if not policy_rows.empty:
        for keys, grp in policy_rows.groupby(CONTEXT_COLS + ["layer"], dropna=False):
            methods = grp["selected_method"].astype(str)
            counts = methods.value_counts()
            total = int(counts.sum())
            entropy, normalized_entropy, effective = entropy_from_counts(counts)
            rec = dict(zip(CONTEXT_COLS + ["layer"], keys))
            rec.update(
                {
                    "num_threshold_settings": total,
                    "dominant_method": counts.index[0] if total else None,
                    "dominant_method_display": method_display(counts.index[0]) if total else None,
                    "dominant_method_count": int(counts.iloc[0]) if total else 0,
                    "dominant_method_share": float(counts.iloc[0] / total) if total else np.nan,
                    "num_methods_seen": int(counts.size),
                    "method_entropy": entropy,
                    "normalized_method_entropy": normalized_entropy,
                    "effective_num_methods": effective,
                    "method_frequency_json": json.dumps(counts.to_dict(), default=str),
                }
            )
            layer_rows.append(rec)
    layer = pd.DataFrame(layer_rows)

    if not layer.empty:
        layer_agg = (
            layer.groupby(CONTEXT_COLS, dropna=False)
            .agg(
                mean_layer_dominant_share=("dominant_method_share", "mean"),
                min_layer_dominant_share=("dominant_method_share", "min"),
                mean_layer_entropy=("method_entropy", "mean"),
                mean_effective_methods=("effective_num_methods", "mean"),
                unstable_layer_count=("dominant_method_share", lambda s: int((pd.to_numeric(s, errors="coerce") < 0.6).sum())),
            )
            .reset_index()
        )
        summary = summary.merge(layer_agg, on=CONTEXT_COLS, how="left")
    return summary, layer


def threshold_response_table(hybrid: pd.DataFrame) -> pd.DataFrame:
    group_cols = CONTEXT_COLS + PARAM_COLS
    agg = (
        hybrid.groupby(group_cols, dropna=False)
        .agg(
            stacks=("stack_id", "count"),
            accuracy_delta_pp=("accuracy_delta_pp", "mean"),
            flops_reduction_pct=("flops_reduction_pct", "mean"),
            params_reduction_pct=("params_reduction_pct", "mean"),
            time_sec=("time_sec", "mean"),
            analysis_score=("analysis_score", "mean"),
            methods_used=("methods_used", lambda s: " + ".join(pd.Series(s).dropna().astype(str).drop_duplicates().head(4))),
        )
        .reset_index()
    )
    return agg


def parameter_sensitivity_curves(hybrid: pd.DataFrame, policy_rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate policy and metric response as one threshold parameter varies.

    Each curve point keeps the exact experiment context fixed
    (objective/dataset/model/scope/ratio), fixes one threshold value, and
    averages over the other threshold-grid dimensions. This makes the resulting
    plots interpretable as "when this setting changes, how much do policies and
    metrics move on average?"
    """
    rows = []
    if hybrid.empty:
        return pd.DataFrame()
    for param in PARAM_COLS:
        group_cols = CONTEXT_COLS + [param]
        for keys, grp in hybrid.groupby(group_cols, dropna=False):
            signatures = grp["policy_signature"].fillna("").astype(str)
            counts = signatures.value_counts(dropna=False)
            n = int(len(grp))
            rec = dict(zip(group_cols, keys))
            rec.update(
                {
                    "parameter": param,
                    "parameter_value": rec[param],
                    "num_settings_at_value": n,
                    "num_unique_policy_signatures": int(counts.size),
                    "modal_policy_share_at_value": float(counts.iloc[0] / n) if n and not counts.empty else np.nan,
                    "accuracy_delta_pp_mean": float(pd.to_numeric(grp["accuracy_delta_pp"], errors="coerce").mean()),
                    "flops_reduction_pct_mean": float(pd.to_numeric(grp["flops_reduction_pct"], errors="coerce").mean()),
                    "params_reduction_pct_mean": float(pd.to_numeric(grp["params_reduction_pct"], errors="coerce").mean()),
                    "time_sec_mean": float(pd.to_numeric(grp["time_sec"], errors="coerce").mean()),
                    "analysis_score_mean": float(pd.to_numeric(grp["analysis_score"], errors="coerce").mean()),
                }
            )
            rows.append(rec)
    curves = pd.DataFrame(rows)
    if curves.empty or policy_rows.empty:
        curves["mean_layer_dominant_share_at_value"] = np.nan
        return curves

    layer_rows = []
    for param in PARAM_COLS:
        group_cols = CONTEXT_COLS + [param, "layer"]
        for keys, grp in policy_rows.groupby(group_cols, dropna=False):
            methods = grp["selected_method"].astype(str)
            counts = methods.value_counts()
            total = int(counts.sum())
            rec = dict(zip(group_cols, keys))
            rec.update(
                {
                    "parameter": param,
                    "parameter_value": rec[param],
                    "layer_dominant_share": float(counts.iloc[0] / total) if total else np.nan,
                }
            )
            layer_rows.append(rec)
    layer_curve = pd.DataFrame(layer_rows)
    if layer_curve.empty:
        curves["mean_layer_dominant_share_at_value"] = np.nan
        return curves
    layer_agg = (
        layer_curve.groupby(CONTEXT_COLS + ["parameter", "parameter_value"], dropna=False)
        .agg(mean_layer_dominant_share_at_value=("layer_dominant_share", "mean"))
        .reset_index()
    )
    curves = curves.merge(layer_agg, on=CONTEXT_COLS + ["parameter", "parameter_value"], how="left")
    return curves


def ols_effects(frame: pd.DataFrame, responses: list[str]) -> pd.DataFrame:
    rows = []
    predictors = ["variance_threshold", "spearman_threshold", "jaccard_threshold", "ratio"]
    cats = ["objective", "dataset", "model", "scope"]
    base = frame.copy()
    X_parts = [pd.Series(1.0, index=base.index, name="Intercept")]
    for p in predictors:
        X_parts.append(pd.to_numeric(base[p], errors="coerce").rename(p))
    for c in cats:
        dummies = pd.get_dummies(base[c].astype(str), prefix=c, drop_first=True, dtype=float)
        X_parts.extend([dummies[col] for col in dummies.columns])
    X = pd.concat(X_parts, axis=1)
    for response in responses:
        y = pd.to_numeric(base[response], errors="coerce")
        valid = y.notna() & X.notna().all(axis=1)
        if int(valid.sum()) < X.shape[1] + 2:
            continue
        xv = X.loc[valid].astype(float)
        yv = y.loc[valid].astype(float)
        xmat = xv.values.astype(float)
        yvec = yv.values.astype(float)
        coef, *_ = np.linalg.lstsq(xmat, yvec, rcond=None)
        pred = xv.values @ coef
        resid = yv.values - pred
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((yv.values - yv.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
        n_obs, p_terms = xmat.shape
        dof = max(0, n_obs - p_terms)
        sigma2 = ss_res / dof if dof > 0 else np.nan
        try:
            cov = sigma2 * np.linalg.pinv(xmat.T @ xmat)
            se = np.sqrt(np.diag(cov))
        except Exception:
            se = np.full_like(coef, np.nan, dtype=float)
        tvals = coef / se
        try:
            from scipy import stats

            pvals = 2.0 * stats.t.sf(np.abs(tvals), df=dof)
            crit = stats.t.ppf(0.975, df=dof)
        except Exception:
            pvals = np.full_like(coef, np.nan, dtype=float)
            crit = np.nan
        ci_low = coef - crit * se if np.isfinite(crit) else np.full_like(coef, np.nan, dtype=float)
        ci_high = coef + crit * se if np.isfinite(crit) else np.full_like(coef, np.nan, dtype=float)
        for name, value in zip(xv.columns, coef):
            i = list(xv.columns).index(name)
            rows.append(
                {
                    "response": response,
                    "term": name,
                    "coef": float(value),
                    "std_error": float(se[i]) if np.isfinite(se[i]) else np.nan,
                    "t_stat": float(tvals[i]) if np.isfinite(tvals[i]) else np.nan,
                    "p_value": float(pvals[i]) if np.isfinite(pvals[i]) else np.nan,
                    "ci95_low": float(ci_low[i]) if np.isfinite(ci_low[i]) else np.nan,
                    "ci95_high": float(ci_high[i]) if np.isfinite(ci_high[i]) else np.nan,
                    "r_squared": r2,
                    "n": int(valid.sum()),
                    "dof": int(dof),
                    "model": "y = beta0 + beta1*variance_threshold + beta2*spearman_threshold + beta3*jaccard_threshold + beta4*ratio + fixed-effect controls(objective,dataset,model,scope) + error",
                }
            )
    return pd.DataFrame(rows)


def standardized_threshold_effects(frame: pd.DataFrame, responses: list[str]) -> pd.DataFrame:
    rows = []
    predictors = ["variance_threshold", "spearman_threshold", "jaccard_threshold", "ratio"]
    base = frame.copy()
    X = pd.DataFrame(index=base.index)
    for p in predictors:
        s = pd.to_numeric(base[p], errors="coerce")
        std = s.std(ddof=0)
        X[p] = (s - s.mean()) / std if np.isfinite(std) and std > 1e-12 else 0.0
    for response in responses:
        y = pd.to_numeric(base[response], errors="coerce")
        ystd = y.std(ddof=0)
        if not np.isfinite(ystd) or ystd <= 1e-12:
            continue
        yz = (y - y.mean()) / ystd
        valid = yz.notna() & X.notna().all(axis=1)
        if int(valid.sum()) < len(predictors) + 2:
            continue
        xv = pd.concat([pd.Series(1.0, index=X.index, name="Intercept"), X], axis=1).loc[valid]
        xmat = xv.values.astype(float)
        yvec = yz.loc[valid].values.astype(float)
        coef, *_ = np.linalg.lstsq(xmat, yvec, rcond=None)
        pred = xmat @ coef
        resid = yvec - pred
        ss_res = float(np.sum(resid**2))
        n_obs, p_terms = xmat.shape
        dof = max(0, n_obs - p_terms)
        sigma2 = ss_res / dof if dof > 0 else np.nan
        try:
            cov = sigma2 * np.linalg.pinv(xmat.T @ xmat)
            se = np.sqrt(np.diag(cov))
        except Exception:
            se = np.full_like(coef, np.nan, dtype=float)
        tvals = coef / se
        try:
            from scipy import stats

            pvals = 2.0 * stats.t.sf(np.abs(tvals), df=dof)
            crit = stats.t.ppf(0.975, df=dof)
        except Exception:
            pvals = np.full_like(coef, np.nan, dtype=float)
            crit = np.nan
        ci_low = coef - crit * se if np.isfinite(crit) else np.full_like(coef, np.nan, dtype=float)
        ci_high = coef + crit * se if np.isfinite(crit) else np.full_like(coef, np.nan, dtype=float)
        for name, value in zip(xv.columns, coef):
            if name == "Intercept":
                continue
            i = list(xv.columns).index(name)
            rows.append(
                {
                    "response": response,
                    "term": name,
                    "standardized_coef": float(value),
                    "std_error": float(se[i]) if np.isfinite(se[i]) else np.nan,
                    "t_stat": float(tvals[i]) if np.isfinite(tvals[i]) else np.nan,
                    "p_value": float(pvals[i]) if np.isfinite(pvals[i]) else np.nan,
                    "ci95_low": float(ci_low[i]) if np.isfinite(ci_low[i]) else np.nan,
                    "ci95_high": float(ci_high[i]) if np.isfinite(ci_high[i]) else np.nan,
                    "n": int(valid.sum()),
                    "dof": int(dof),
                    "model": "z(y) = beta0 + beta1*z(variance_threshold) + beta2*z(spearman_threshold) + beta3*z(jaccard_threshold) + beta4*z(ratio) + error",
                }
            )
    return pd.DataFrame(rows)


def plot_stability(summary: pd.DataFrame, plot_dir: Path) -> list[dict[str, str]]:
    apply_thesis_style()
    manifests = []
    if summary.empty:
        return manifests
    work = summary.sort_values(["objective_label", "dataset", "model", "scope", "ratio"]).copy()
    labels = [context_label(r, width=30) for _, r in work.iterrows()]
    x = np.arange(len(work))
    fig, ax = plt.subplots(figsize=(max(14, 0.38 * len(work)), 6.4))
    ax.bar(x - 0.18, work["modal_policy_share"], width=0.36, label="Full-policy repeat rate", color="#2563EB", alpha=0.92)
    if "mean_layer_dominant_share" in work.columns:
        ax.bar(x + 0.18, work["mean_layer_dominant_share"], width=0.36, label="Mean layer-method repeat rate", color="#0E9F6E", alpha=0.92)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Stability score")
    ax.set_title("Policy stability across Algorithm-2 threshold settings", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=6.8)
    prettify_axis(ax)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.text(
        0.01,
        0.98,
        "Full-policy stability is stricter: every layer must repeat the same method.\nLayer-method stability asks whether individual layers repeatedly prefer the same method.",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.95),
    )
    fig.tight_layout()
    out = plot_dir / "policy_stability_by_context.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    manifests.append({"plot": str(out), "description": "Full-policy and layerwise stability by exact context"})

    metric_specs = [
        ("accuracy_delta_range", "Accuracy delta range across thresholds (pp)", "#10B981"),
        ("flops_reduction_range", "FLOPs reduction range across thresholds (%)", "#2563EB"),
        ("time_sec_range", "Pruning-time range across thresholds (s)", "#F97316"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 7.5))
    for ax, (col, title, color) in zip(axes, metric_specs):
        top = work.assign(_metric=pd.to_numeric(work[col], errors="coerce")).sort_values("_metric", ascending=False).head(18)
        y = np.arange(len(top))
        bars = ax.barh(y, top["_metric"], color=color, alpha=0.86)
        ax.set_title(title, pad=10)
        ax.set_yticks(y)
        ax.set_yticklabels([context_label(r, width=31) for _, r in top.iterrows()], fontsize=7)
        ax.invert_yaxis()
        ax.grid(True, axis="x", color="#CBD5E1", alpha=0.35, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        try:
            ax.bar_label(bars, labels=[f"{v:.1f}" for v in top["_metric"]], padding=3, fontsize=7.5, color="#111827")
        except Exception:
            pass
    fig.suptitle("Most threshold-sensitive contexts by metric range", y=1.01, fontsize=15, fontweight="semibold")
    fig.tight_layout()
    out = plot_dir / "metric_sensitivity_ranges_by_context.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    manifests.append({"plot": str(out), "description": "Accuracy, FLOPs, and time ranges under threshold changes"})

    if "mean_layer_dominant_share" not in summary.columns:
        return manifests
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    scatter = ax.scatter(
        pd.to_numeric(summary["mean_layer_dominant_share"], errors="coerce"),
        pd.to_numeric(summary["analysis_score_mean"], errors="coerce"),
        s=70 + 260 * norm_high(summary["num_threshold_settings"]),
        c=pd.to_numeric(summary["accuracy_delta_range"], errors="coerce"),
        cmap="mako_r" if "mako_r" in plt.colormaps() else "viridis_r",
        alpha=0.82,
        edgecolors="#111827",
        linewidths=0.45,
    )
    ax.set_xlabel("Mean layer-method stability")
    ax.set_ylabel("Mean objective-aware score")
    ax.set_title("Policy stability vs final stack quality", pad=12)
    prettify_axis(ax)
    cbar = fig.colorbar(scatter, ax=ax, label="Accuracy sensitivity range (pp)")
    cbar.outline.set_visible(False)
    ax.text(
        0.02,
        0.04,
        "Larger markers = more threshold settings tested",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CBD5E1", alpha=0.9),
    )
    fig.tight_layout()
    out = plot_dir / "stability_vs_quality.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    manifests.append({"plot": str(out), "description": "Trade-off between stable policies and stack quality"})
    return manifests


def plot_layer_heatmaps(layer: pd.DataFrame, summary: pd.DataFrame, plot_dir: Path, max_contexts: int = 24) -> list[dict[str, str]]:
    manifests = []
    if layer.empty or summary.empty:
        return manifests
    contexts = summary.sort_values(["modal_policy_share", "mean_layer_dominant_share", "analysis_score_mean"], ascending=[True, True, False]).head(max_contexts)
    for _, ctx in contexts.iterrows():
        mask = pd.Series(True, index=layer.index)
        for col in CONTEXT_COLS:
            mask &= layer[col].astype(str).eq(str(ctx[col]))
        sub = layer[mask].copy()
        if sub.empty:
            continue
        sub["layer_order"] = np.arange(len(sub))
        fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(sub)), 3.8))
        vals = pd.to_numeric(sub["dominant_method_share"], errors="coerce").to_numpy()[None, :]
        im = ax.imshow(vals, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
        ax.set_yticks([0])
        ax.set_yticklabels(["stability"])
        ax.set_xticks(np.arange(len(sub)))
        ax.set_xticklabels(sub["layer"].astype(str), rotation=60, ha="right", fontsize=7)
        for i, row in enumerate(sub.itertuples(index=False)):
            ax.text(i, 0, f"{getattr(row, 'dominant_method_display')}\n{getattr(row, 'dominant_method_share'):.2f}", ha="center", va="center", fontsize=6)
        title = f"Layerwise policy stability | {ctx.objective_label} | {ctx.dataset} | {ctx.model} | {ctx.scope} | r={float(ctx.ratio):g}"
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="Dominant method share")
        fig.tight_layout()
        out = plot_dir / f"layer_policy_stability_{safe_slug(ctx.objective)}_{safe_slug(ctx.dataset)}_{safe_slug(ctx.model)}_{safe_slug(ctx.scope)}_r{safe_slug(ctx.ratio)}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        manifests.append({"plot": str(out), "description": title})
    return manifests


def plot_threshold_response(response: pd.DataFrame, plot_dir: Path, max_contexts: int = 36) -> list[dict[str, str]]:
    apply_thesis_style()
    manifests = []
    if response.empty:
        return manifests
    response = response.copy()
    for col in PARAM_COLS + ["accuracy_delta_pp", "flops_reduction_pct", "time_sec"]:
        response[col] = pd.to_numeric(response[col], errors="coerce")
    for metric, label, cmap, sensitivity_col in [
        ("accuracy_delta_pp", "Accuracy delta (pp)", "RdYlGn", "accuracy_delta_pp"),
        ("flops_reduction_pct", "FLOPs reduction (%)", "YlGnBu", "flops_reduction_pct"),
        ("time_sec", "Pruning time (s)", "magma_r", "time_sec"),
    ]:
        metric_ranges = []
        for _, ctx in response[CONTEXT_COLS].drop_duplicates().iterrows():
            mask = pd.Series(True, index=response.index)
            for c in CONTEXT_COLS:
                mask &= response[c].astype(str).eq(str(ctx[c]))
            vals = pd.to_numeric(response.loc[mask, sensitivity_col], errors="coerce")
            rec = ctx.to_dict()
            rec["_range"] = float(vals.max() - vals.min()) if vals.notna().any() else np.nan
            metric_ranges.append(rec)
        contexts = pd.DataFrame(metric_ranges).sort_values("_range", ascending=False).head(min(max_contexts, 16))
        n = len(contexts)
        if n == 0:
            continue
        cols = 4
        rows = int(math.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.25, rows * 2.85), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        all_vals = []
        for _, ctx in contexts.iterrows():
            mask = pd.Series(True, index=response.index)
            for c in CONTEXT_COLS:
                mask &= response[c].astype(str).eq(str(ctx[c]))
            all_vals.extend(pd.to_numeric(response.loc[mask, metric], errors="coerce").dropna().tolist())
        vmin = float(np.nanmin(all_vals)) if all_vals else None
        vmax = float(np.nanmax(all_vals)) if all_vals else None
        last_im = None
        for ax, (_, ctx) in zip(axes.ravel(), contexts.iterrows()):
            mask = pd.Series(True, index=response.index)
            for c in CONTEXT_COLS:
                mask &= response[c].astype(str).eq(str(ctx[c]))
            sub = response[mask]
            piv = sub.pivot_table(index="jaccard_threshold", columns="spearman_threshold", values=metric, aggfunc="mean")
            if piv.empty:
                continue
            ax.axis("on")
            last_im = ax.imshow(piv.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks(np.arange(len(piv.columns)))
            ax.set_xticklabels([f"{x:g}" for x in piv.columns], fontsize=7)
            ax.set_yticks(np.arange(len(piv.index)))
            ax.set_yticklabels([f"{x:g}" for x in piv.index], fontsize=7)
            ax.set_xlabel("Spearman", fontsize=7.5)
            ax.set_ylabel("Jaccard", fontsize=7.5)
            ax.set_title(f"{ctx.dataset} {ctx.model}\n{ctx.objective_label} | {ctx.scope} | r={float(ctx.ratio):g}", fontsize=8.2)
            for i in range(piv.shape[0]):
                for j in range(piv.shape[1]):
                    val = piv.values[i, j]
                    text_color = "white" if pd.notna(val) and vmin is not None and vmax is not None and (val - vmin) / max(vmax - vmin, 1e-9) < 0.28 else "#111827"
                    ax.text(j, i, "" if pd.isna(val) else f"{val:.1f}", ha="center", va="center", fontsize=7.2, color=text_color)
        fig.suptitle(f"Threshold response: {label}", y=0.995, fontsize=15, fontweight="semibold")
        if last_im is not None:
            cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.74, pad=0.012)
            cbar.set_label(label)
            cbar.outline.set_visible(False)
        fig.tight_layout(rect=(0, 0, 0.96, 0.97))
        out = plot_dir / f"threshold_response_{safe_slug(metric)}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        manifests.append({"plot": str(out), "description": f"Threshold response for {label}"})
    return manifests


def plot_parameter_curves(curves: pd.DataFrame, plot_dir: Path) -> list[dict[str, str]]:
    apply_thesis_style()
    manifests = []
    if curves.empty:
        return manifests
    work = curves.copy()
    for col in ["parameter_value", "modal_policy_share_at_value", "mean_layer_dominant_share_at_value", "accuracy_delta_pp_mean", "flops_reduction_pct_mean", "time_sec_mean", "analysis_score_mean"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    objective_order = list(work["objective_label"].dropna().drop_duplicates())
    fallback = ["#2563EB", "#0E9F6E", "#E76F51", "#7C3AED", "#DC2626"]
    colors = {obj: THESIS_COLORS.get(obj, fallback[i % len(fallback)]) for i, obj in enumerate(objective_order)}

    stability_metrics = [
        ("modal_policy_share_at_value", "Full-policy stability"),
        ("mean_layer_dominant_share_at_value", "Mean layer-method stability"),
    ]
    fig, axes = plt.subplots(len(stability_metrics), len(PARAM_COLS), figsize=(16, 7.4), squeeze=False)
    for row_i, (metric, ylabel) in enumerate(stability_metrics):
        for col_i, param in enumerate(PARAM_COLS):
            ax = axes[row_i][col_i]
            sub = work[work["parameter"].eq(param)].dropna(subset=["parameter_value", metric])
            for objective, grp in sub.groupby("objective_label"):
                agg = grp.groupby("parameter_value", dropna=False)[metric].agg(["mean", "std", "count"]).reset_index().sort_values("parameter_value")
                ax.plot(agg["parameter_value"], agg["mean"], marker="o", markersize=5, linewidth=2.1, label=objective, color=colors.get(objective))
                if (agg["count"] > 1).any():
                    lo = agg["mean"] - agg["std"].fillna(0)
                    hi = agg["mean"] + agg["std"].fillna(0)
                    ax.fill_between(agg["parameter_value"].to_numpy(float), lo.to_numpy(float), hi.to_numpy(float), alpha=0.10, color=colors.get(objective))
            ax.set_title(param.replace("_", " ").title(), pad=8)
            ax.set_xlabel("Threshold value")
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 1.05)
            prettify_axis(ax)
            if row_i == 0 and col_i == len(PARAM_COLS) - 1:
                ax.legend(fontsize=8.2, loc="best", frameon=True, edgecolor="#CBD5E1")
    fig.suptitle("How Algorithm-2 thresholds influence discovered policies", y=1.02, fontsize=15, fontweight="semibold")
    fig.tight_layout()
    out = plot_dir / "parameter_curves_policy_stability.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    manifests.append({"plot": str(out), "description": "Line curves showing policy stability as each threshold changes"})

    metric_specs = [
        ("accuracy_delta_pp_mean", "Accuracy delta (pp)"),
        ("flops_reduction_pct_mean", "FLOPs reduction (%)"),
        ("time_sec_mean", "Pruning time (s)"),
        ("analysis_score_mean", "Objective-aware score"),
    ]
    fig, axes = plt.subplots(len(metric_specs), len(PARAM_COLS), figsize=(16, 11.3), squeeze=False)
    for row_i, (metric, ylabel) in enumerate(metric_specs):
        for col_i, param in enumerate(PARAM_COLS):
            ax = axes[row_i][col_i]
            sub = work[work["parameter"].eq(param)].dropna(subset=["parameter_value", metric])
            for objective, grp in sub.groupby("objective_label"):
                agg = grp.groupby("parameter_value", dropna=False)[metric].agg(["mean", "std", "count"]).reset_index().sort_values("parameter_value")
                ax.plot(agg["parameter_value"], agg["mean"], marker="o", markersize=5, linewidth=2.1, label=objective, color=colors.get(objective))
                if (agg["count"] > 1).any():
                    lo = agg["mean"] - agg["std"].fillna(0)
                    hi = agg["mean"] + agg["std"].fillna(0)
                    ax.fill_between(agg["parameter_value"].to_numpy(float), lo.to_numpy(float), hi.to_numpy(float), alpha=0.10, color=colors.get(objective))
            ax.set_title(param.replace("_", " ").title(), pad=8)
            ax.set_xlabel("Threshold value")
            ax.set_ylabel(ylabel)
            prettify_axis(ax)
            if row_i == 0 and col_i == len(PARAM_COLS) - 1:
                ax.legend(fontsize=8.2, loc="best", frameon=True, edgecolor="#CBD5E1")
    fig.suptitle("How Algorithm-2 thresholds influence final stack quality", y=1.01, fontsize=15, fontweight="semibold")
    fig.tight_layout()
    out = plot_dir / "parameter_curves_stack_quality.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    manifests.append({"plot": str(out), "description": "Line curves showing stack metrics as each threshold changes"})
    return manifests


def plot_standardized_effects(effects: pd.DataFrame, plot_dir: Path) -> list[dict[str, str]]:
    apply_thesis_style()
    manifests = []
    if effects.empty:
        return manifests
    work = effects[effects["term"].isin(PARAM_COLS + ["ratio"])].copy()
    if work.empty:
        return manifests
    work["standardized_coef"] = pd.to_numeric(work["standardized_coef"], errors="coerce")
    work["ci95_low"] = pd.to_numeric(work.get("ci95_low", np.nan), errors="coerce")
    work["ci95_high"] = pd.to_numeric(work.get("ci95_high", np.nan), errors="coerce")
    work["p_value"] = pd.to_numeric(work.get("p_value", np.nan), errors="coerce")
    responses = list(work["response"].drop_duplicates())
    terms = PARAM_COLS + ["ratio"]
    fig, axes = plt.subplots(len(responses), 1, figsize=(10.5, max(4, 2.2 * len(responses))), squeeze=False)
    for ax, response in zip(axes.ravel(), responses):
        sub = work[work["response"].eq(response)].set_index("term").reindex(terms).reset_index()
        colors = ["#10B981" if v >= 0 else "#EF4444" for v in sub["standardized_coef"].fillna(0)]
        labels = sub["term"].str.replace("_", " ").str.title()
        bars = ax.barh(labels, sub["standardized_coef"], color=colors, alpha=0.88)
        if {"ci95_low", "ci95_high"}.issubset(sub.columns):
            xerr = np.vstack(
                [
                    (sub["standardized_coef"] - sub["ci95_low"]).clip(lower=0).fillna(0),
                    (sub["ci95_high"] - sub["standardized_coef"]).clip(lower=0).fillna(0),
                ]
            )
            ax.errorbar(sub["standardized_coef"], labels, xerr=xerr, fmt="none", ecolor="#334155", elinewidth=1, capsize=3, alpha=0.85)
        ax.axvline(0, color="#111827", linewidth=1)
        ax.set_title(response.replace("_", " "), pad=8)
        ax.set_xlabel("Standardized coefficient")
        ax.grid(True, axis="x", color="#CBD5E1", alpha=0.35, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, coef, pval in zip(bars, sub["standardized_coef"], sub["p_value"]):
            if pd.isna(coef):
                continue
            stars = "***" if pd.notna(pval) and pval < 0.001 else "**" if pd.notna(pval) and pval < 0.01 else "*" if pd.notna(pval) and pval < 0.05 else ""
            ha = "left" if coef >= 0 else "right"
            pad = 0.012 if coef >= 0 else -0.012
            ax.text(coef + pad, bar.get_y() + bar.get_height() / 2, f"{coef:.2f}{stars}", va="center", ha=ha, fontsize=8.5, color="#111827")
    fig.suptitle("Pooled standardized parameter effects with 95% confidence intervals", y=1.01, fontsize=15, fontweight="semibold")
    fig.text(0.995, 0.004, "* p<0.05, ** p<0.01, *** p<0.001", ha="right", fontsize=8.5, color="#475569")
    fig.tight_layout()
    out = plot_dir / "pooled_standardized_parameter_effects.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    manifests.append({"plot": str(out), "description": "Pooled model effect sizes for thresholds and prune ratio"})
    return manifests


def write_notebook(out_dir: Path, notebook_path: Path) -> None:
    try:
        rel = out_dir.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        rel = out_dir.as_posix()
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# LFPC Policy-Parameter Sensitivity Analysis v2\n",
                "\n",
                "This notebook rebuilds the sensitivity section from the latest experiment registry. It separates exact contexts and measures policy stability, not only metric movement.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "import pandas as pd\n",
                "from IPython.display import display, Markdown, Image\n",
                "\n",
                f"REPORT_DIR = Path('{rel}')\n",
                "TABLE_DIR = REPORT_DIR / 'tables'\n",
                "PLOT_DIR = REPORT_DIR / 'plots'\n",
                "\n",
                "def read_table(name):\n",
                "    path = TABLE_DIR / name\n",
                "    if not path.exists() or path.stat().st_size == 0:\n",
                "        return pd.DataFrame()\n",
                "    try:\n",
                "        return pd.read_csv(path)\n",
                "    except pd.errors.EmptyDataError:\n",
                "        return pd.DataFrame()\n",
                "\n",
                "tables = {\n",
                "    'Context policy stability': read_table('policy_stability_by_context.csv'),\n",
                "    'Layerwise policy stability': read_table('layerwise_policy_stability.csv'),\n",
                "    'Threshold response': read_table('threshold_response_by_context.csv'),\n",
                "    'Parameter sensitivity curves': read_table('parameter_sensitivity_curves.csv'),\n",
                "    'Pooled parameter effects': read_table('pooled_parameter_effects.csv'),\n",
                "    'Pooled standardized parameter effects': read_table('pooled_standardized_parameter_effects.csv'),\n",
                "    'Report manifest': read_table('plot_manifest.csv'),\n",
                "}\n",
                "for title, df in tables.items():\n",
                "    display(Markdown(f'## {title}'))\n",
                "    display(df.head(80) if not df.empty else Markdown('No rows available.'))\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "display(Markdown('## Sensitivity Plots'))\n",
                "manifest = tables['Report manifest']\n",
                "for _, row in manifest.iterrows():\n",
                "    path = Path(row['plot'])\n",
                "    if path.exists():\n",
                "        display(Markdown(f\"### {row.get('description', path.name)}\"))\n",
                "        display(Image(filename=str(path)))\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "display(Markdown('## Least Stable Context Drill-Down'))\n",
                "summary = tables['Context policy stability']\n",
                "layer = tables['Layerwise policy stability']\n",
                "if not summary.empty:\n",
                "    cols = ['objective_label','dataset','model','scope','ratio','num_threshold_settings','modal_policy_share','mean_layer_dominant_share','accuracy_delta_range','flops_reduction_range','time_sec_range','best_stack_id','best_methods_used']\n",
                "    display(summary[[c for c in cols if c in summary.columns]].sort_values(['modal_policy_share','mean_layer_dominant_share']).head(20))\n",
                "if not layer.empty:\n",
                "    cols = ['objective_label','dataset','model','scope','ratio','layer','dominant_method_display','dominant_method_share','num_methods_seen','method_frequency_json']\n",
                "    display(layer[[c for c in cols if c in layer.columns]].sort_values(['dominant_method_share','num_methods_seen'], ascending=[True, False]).head(80))\n",
            ],
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Regression Model Used\n",
                "\n",
                "For the unstandardized pooled model, each response is fitted as:\n",
                "\n",
                "$$y_i = \\beta_0 + \\beta_1 v_i + \\beta_2 s_i + \\beta_3 j_i + \\beta_4 r_i + \\gamma_o O_i + \\gamma_d D_i + \\gamma_m M_i + \\gamma_q Q_i + \\epsilon_i$$\n",
                "\n",
                "where $v_i$ is the variance threshold, $s_i$ the Spearman threshold, $j_i$ the Jaccard threshold, $r_i$ the prune ratio, and the $\\gamma$ terms are fixed-effect controls for objective, dataset, model, and scope.\n",
                "\n",
                "For the standardized coefficient plot/table, the same threshold predictors are z-scored and fitted as:\n",
                "\n",
                "$$z(y_i) = \\beta_0 + \\beta_1 z(v_i) + \\beta_2 z(s_i) + \\beta_3 z(j_i) + \\beta_4 z(r_i) + \\epsilon_i$$\n",
                "\n",
                "The standardized model is intended for effect-size comparison; the fixed-effect pooled model is intended for adjusted inference.\n",
            ],
        },
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "pygments_lexer": "ipython3"}}, "nbformat": 4, "nbformat_minor": 5}
    notebook_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")


@dataclass
class BuildResult:
    out_dir: Path
    hybrid_rows: int
    policy_rows: int
    context_rows: int
    layer_rows: int
    plots: int


def build_report(
    registry_path: Path,
    out_dir: Path,
    v4_metrics_path: Path | None = DEFAULT_V4_METRICS,
    max_layer_heatmaps: int = 24,
    write_analysis_notebook: bool = True,
) -> BuildResult:
    table_dir = out_dir / "tables"
    plot_dir = out_dir / "plots"
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    hybrid = latest_hybrid_contexts(registry_path, v4_metrics_path)
    policy_rows = load_policy_rows(hybrid, PROJECT_ROOT)
    hybrid = build_policy_signatures(hybrid, policy_rows)

    summary, layer = stability_tables(hybrid, policy_rows)
    response = threshold_response_table(hybrid)
    curves = parameter_sensitivity_curves(hybrid, policy_rows)
    effects = ols_effects(
        response,
        ["accuracy_delta_pp", "flops_reduction_pct", "params_reduction_pct", "time_sec", "analysis_score"],
    )
    standardized_effects = standardized_threshold_effects(
        response,
        ["accuracy_delta_pp", "flops_reduction_pct", "params_reduction_pct", "time_sec", "analysis_score"],
    )

    hybrid.to_csv(table_dir / "latest_hybrid_threshold_contexts.csv", index=False)
    policy_rows.to_csv(table_dir / "policy_rows_from_latest_runs.csv", index=False)
    summary.to_csv(table_dir / "policy_stability_by_context.csv", index=False)
    layer.to_csv(table_dir / "layerwise_policy_stability.csv", index=False)
    response.to_csv(table_dir / "threshold_response_by_context.csv", index=False)
    curves.to_csv(table_dir / "parameter_sensitivity_curves.csv", index=False)
    effects.to_csv(table_dir / "pooled_parameter_effects.csv", index=False)
    standardized_effects.to_csv(table_dir / "pooled_standardized_parameter_effects.csv", index=False)

    manifest_rows = []
    manifest_rows.extend(plot_parameter_curves(curves, plot_dir))
    manifest_rows.extend(plot_stability(summary, plot_dir))
    manifest_rows.extend(plot_standardized_effects(standardized_effects, plot_dir))
    manifest_rows.extend(plot_layer_heatmaps(layer, summary, plot_dir, max_contexts=max_layer_heatmaps))
    manifest_rows.extend(plot_threshold_response(response, plot_dir))
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(table_dir / "plot_manifest.csv", index=False)

    summary_payload = {
        "hybrid_threshold_context_rows": int(len(hybrid)),
        "policy_layer_rows": int(len(policy_rows)),
        "context_stability_rows": int(len(summary)),
        "layerwise_stability_rows": int(len(layer)),
        "plot_count": int(len(manifest_rows)),
        "registry_path": str(registry_path),
        "v4_metrics_path": str(v4_metrics_path) if v4_metrics_path else None,
    }
    (out_dir / "policy_parameter_sensitivity_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    if write_analysis_notebook:
        write_notebook(out_dir, PROJECT_ROOT / "policy_parameter_sensitivity_analysis_v2.ipynb")
    return BuildResult(out_dir, len(hybrid), len(policy_rows), len(summary), len(layer), len(manifest_rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--v4-metrics", type=Path, default=DEFAULT_V4_METRICS)
    parser.add_argument("--max-layer-heatmaps", type=int, default=24)
    args = parser.parse_args()
    result = build_report(args.registry, args.out_dir, args.v4_metrics, args.max_layer_heatmaps)
    print(json.dumps(result.__dict__ | {"out_dir": str(result.out_dir)}, indent=2, default=str))


if __name__ == "__main__":
    main()
