"""Build a context-safe report from saved pruned model checkpoints.

This v4 builder treats checkpointed models as the source of truth for structural
metrics. CSV/registry artifacts are used only to discover context, stack ids,
method ids, saved model paths, and layerwise policy metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


METHOD_DISPLAY = {
    "l1_norm": "L1",
    "custom_l2": "L2",
    "mean_abs_act": "MeanAct",
    "apoz": "APoZ",
    "custom_entropy": "Entropy",
    "custom_class_entropy": "ClassEntropy",
    "custom_hrank": "HRank",
    "custom_spectral_energy": "Spectral",
    "chip": "CHIP",
    "custom_reprune": "REPrune",
    "custom_tis": "TIS",
    "custom_nisp": "NISP",
    "custom_senpis": "SeNPIS",
    "custom_senpips": "SeNPIPS",
    "custom_thinet": "ThiNet",
    "custom_dcp": "DCP",
    "custom_gfs": "GFS",
    "custom_autodfp": "AutoDFP",
    "custom_gfi_ap": "GFI-AP",
}

CONTEXT_KEYS = ["objective", "dataset", "model", "scope", "ratio"]


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def method_display(method: Any) -> str:
    text = norm_text(method)
    return METHOD_DISPLAY.get(text, text)


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def safe_ratio(value: Any) -> float:
    return round(safe_float(value), 8)


def slug(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", norm_text(value)).strip("_")
    return text or "item"


def parse_listish(value: Any) -> list[str]:
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
    return [x.strip().strip("'\"") for x in re.split(r"[,+|]", value) if x.strip().strip("'\"")]


def resolve_path(value: Any, project_root: Path) -> Optional[Path]:
    text = norm_text(value)
    if not text:
        return None
    text = text.replace("\\", "/")
    direct = Path(text)
    if direct.exists():
        return direct
    for marker in ["activation-based-pruning/", "outputs/lfpc_hybrid/", "saved_models/"]:
        idx = text.lower().find(marker.lower())
        if idx >= 0:
            suffix = text[idx + len("activation-based-pruning/") :] if marker == "activation-based-pruning/" else text[idx:]
            candidate = project_root / Path(suffix)
            if candidate.exists():
                return candidate
    return direct if direct.exists() else None


def dataset_defaults(dataset: Any) -> Tuple[Tuple[int, int, int], int]:
    text = norm_text(dataset).lower()
    if "cat" in text and "dog" in text:
        return (3, 128, 128), 2
    if "cifar-100" in text or "cifar100" in text:
        return (3, 32, 32), 100
    return (3, 32, 32), 10


def canonical_model_name(model: Any) -> str:
    text = norm_text(model).lower().replace("-", "_")
    if "resnet18" in text or text == "resnet":
        return "resnet18"
    if "vgg16" in text or text == "vgg":
        return "vgg16"
    if "mobilenet" in text:
        return "mobilenet_v2"
    return text


def extract_checkpoint_model(payload: Any) -> Tuple[Any, Dict[str, Any], str]:
    if isinstance(payload, dict):
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        for key in ["model", "module"]:
            model = payload.get(key)
            if model is not None and hasattr(model, "parameters"):
                return model, metadata, key
        raise RuntimeError("checkpoint has no serialized model object; cannot recompute structural FLOPs from state_dict alone")
    if hasattr(payload, "parameters"):
        return payload, {}, "model_object"
    raise RuntimeError(f"unsupported checkpoint payload type: {type(payload)!r}")


def hook_flops(model: Any, input_shape: Tuple[int, int, int], device: str = "cpu") -> Tuple[float, float]:
    import torch
    import torch.nn as nn

    model = model.to(device)
    was_training = bool(getattr(model, "training", False))
    model.eval()
    dummy = torch.randn(1, *input_shape, device=device)
    total_flops = 0.0
    hooks = []

    def conv_hook(module: Any, _inputs: Any, output: Any) -> None:
        nonlocal total_flops
        if not hasattr(output, "shape") or len(output.shape) < 4:
            return
        batch, out_channels, out_h, out_w = [int(v) for v in output.shape[:4]]
        kh, kw = module.kernel_size
        kernel_ops = (int(module.in_channels) // int(module.groups)) * int(kh) * int(kw)
        total_flops += float(batch * out_channels * out_h * out_w * kernel_ops)

    def linear_hook(module: Any, _inputs: Any, output: Any) -> None:
        nonlocal total_flops
        if hasattr(output, "numel"):
            total_flops += float(output.numel() * int(module.in_features))

    try:
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                hooks.append(module.register_forward_hook(conv_hook))
            elif isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(linear_hook))
        with torch.no_grad():
            model(dummy)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)

    params = float(sum(p.numel() for p in model.parameters()))
    if not math.isfinite(total_flops) or total_flops <= 0:
        raise RuntimeError(f"invalid checkpoint FLOPs: {total_flops}")
    if not math.isfinite(params) or params <= 0:
        raise RuntimeError(f"invalid checkpoint params: {params}")
    return float(total_flops), params


@dataclass(frozen=True)
class BaselineKey:
    dataset: str
    model: str
    input_shape: Tuple[int, int, int]
    num_classes: int


def baseline_metrics(key: BaselineKey, device: str, cache: Dict[BaselineKey, Tuple[float, float]]) -> Tuple[float, float]:
    if key in cache:
        return cache[key]
    from reducnn.backends.torch_backend import PyTorchAdapter

    adapter = PyTorchAdapter(
        {
            "input_shape": key.input_shape,
            "num_classes": key.num_classes,
            "device": device,
            "baseline_checkpoint_policy": "off",
        }
    )
    model = adapter.get_model(canonical_model_name(key.model), input_shape=key.input_shape, num_classes=key.num_classes, pretrained=False)
    metrics = hook_flops(model, key.input_shape, device=device)
    cache[key] = metrics
    return metrics


def checkpoint_metrics(row: pd.Series, project_root: Path, device: str, baseline_cache: Dict[BaselineKey, Tuple[float, float]]) -> Dict[str, Any]:
    import torch

    original_path = row.get("checkpoint_path")
    path = resolve_path(original_path, project_root)
    base = row.to_dict()
    base["checkpoint_path_original"] = norm_text(original_path)
    base["checkpoint_path_resolved"] = str(path) if path else ""
    base["checkpoint_exists"] = bool(path and path.exists())
    base["metric_source"] = "checkpoint_model"
    if not path or not path.exists():
        base.update({"metric_status": "missing_checkpoint", "metric_error": "checkpoint path could not be resolved"})
        return base

    try:
        payload = torch.load(path, map_location=device, weights_only=False)
        model, metadata, payload_model_key = extract_checkpoint_model(payload)
        input_shape = tuple(int(x) for x in metadata.get("input_shape", dataset_defaults(row.get("dataset"))[0]))
        num_classes = int(metadata.get("num_classes", dataset_defaults(row.get("dataset"))[1]))
        model_flops, model_params = hook_flops(model, input_shape, device=device)
        bkey = BaselineKey(norm_text(row.get("dataset")), norm_text(row.get("model")), input_shape, num_classes)
        base_flops, base_params = baseline_metrics(bkey, device=device, cache=baseline_cache)
        flops_red = 100.0 * (base_flops - model_flops) / base_flops if base_flops > 0 else math.nan
        params_red = 100.0 * (base_params - model_params) / base_params if base_params > 0 else math.nan
        base.update(
            {
                "metric_status": "ok",
                "metric_error": "",
                "checkpoint_payload_model_key": payload_model_key,
                "checkpoint_metadata_json": json.dumps(metadata, default=str),
                "direct_input_shape": "x".join(str(x) for x in input_shape),
                "direct_num_classes": num_classes,
                "direct_baseline_flops": base_flops,
                "direct_model_flops": model_flops,
                "direct_flops_reduction_pct": flops_red,
                "direct_baseline_params": base_params,
                "direct_model_params": model_params,
                "direct_params_reduction_pct": params_red,
            }
        )
    except Exception as exc:
        base.update({"metric_status": "failed", "metric_error": f"{type(exc).__name__}: {exc}"})
    return base


def add_absolute_metric_units(metrics: pd.DataFrame) -> pd.DataFrame:
    """Add thesis-facing units and explicit provenance to direct metrics."""
    if metrics.empty:
        return metrics.copy()
    out = metrics.copy()
    numeric_cols = [
        "direct_baseline_flops",
        "direct_model_flops",
        "direct_flops_reduction_pct",
        "direct_baseline_params",
        "direct_model_params",
        "direct_params_reduction_pct",
    ]
    for col in numeric_cols:
        if col not in out.columns:
            out[col] = math.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["direct_baseline_gops"] = out["direct_baseline_flops"] / 1e9
    out["direct_model_gops"] = out["direct_model_flops"] / 1e9
    out["direct_removed_gops"] = (out["direct_baseline_flops"] - out["direct_model_flops"]) / 1e9
    out["direct_baseline_params_m"] = out["direct_baseline_params"] / 1e6
    out["direct_model_params_m"] = out["direct_model_params"] / 1e6
    out["direct_removed_params_m"] = (out["direct_baseline_params"] - out["direct_model_params"]) / 1e6
    ok = out.get("metric_status", pd.Series(index=out.index, dtype=str)).astype(str).eq("ok")
    out["absolute_metric_provenance"] = np.where(ok, "checkpoint_direct", "missing")
    out["operation_count_convention"] = "one multiply-accumulate counted as one operation"
    return out


def build_baseline_scale_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return one baseline scale row per dataset/model/input-shape context."""
    if metrics.empty:
        return pd.DataFrame()
    ok = metrics[
        metrics.get("metric_status", pd.Series(index=metrics.index, dtype=str)).astype(str).eq("ok")
    ].copy()
    required = [
        "dataset",
        "model",
        "direct_input_shape",
        "direct_num_classes",
        "direct_baseline_gops",
        "direct_baseline_params_m",
    ]
    if ok.empty or not set(required).issubset(ok.columns):
        return pd.DataFrame()
    grouped = (
        ok.groupby(["dataset", "model", "direct_input_shape", "direct_num_classes"], dropna=False)
        .agg(
            baseline_gops=("direct_baseline_gops", "median"),
            baseline_gops_min=("direct_baseline_gops", "min"),
            baseline_gops_max=("direct_baseline_gops", "max"),
            baseline_params_m=("direct_baseline_params_m", "median"),
            baseline_params_m_min=("direct_baseline_params_m", "min"),
            baseline_params_m_max=("direct_baseline_params_m", "max"),
            profiled_checkpoint_rows=("checkpoint_path_resolved", "count"),
        )
        .reset_index()
    )
    grouped["baseline_gops_invariant"] = np.isclose(
        grouped["baseline_gops_min"], grouped["baseline_gops_max"], rtol=0, atol=1e-12
    )
    grouped["baseline_params_invariant"] = np.isclose(
        grouped["baseline_params_m_min"], grouped["baseline_params_m_max"], rtol=0, atol=1e-12
    )
    grouped["operation_count_convention"] = "one multiply-accumulate counted as one operation"
    grouped["metric_provenance"] = "checkpoint_direct"
    return grouped


def metric_cache_key(row: pd.Series) -> str:
    """Identify one profiled checkpoint inside its exact comparison context."""
    record_type = norm_text(row.get("record_type"))
    report_stack_id = (
        norm_text(row.get("report_stack_id"))
        if record_type == "hybrid"
        else norm_text(row.get("hybrid_report_stack_id"))
    )
    parts = [
        record_type,
        norm_text(row.get("dataset")),
        norm_text(row.get("model")),
        norm_text(row.get("scope")),
        f"{safe_ratio(row.get('ratio')):.8f}",
        report_stack_id,
        norm_text(row.get("method")),
        norm_text(row.get("checkpoint_path", row.get("checkpoint_path_original"))).replace("\\", "/"),
    ]
    return "|".join(parts)


def build_checkpoint_index(top: pd.DataFrame, singular_cache: pd.DataFrame) -> pd.DataFrame:
    rows = []
    top = top.copy()
    top["ratio"] = pd.to_numeric(top["ratio"], errors="coerce")
    for _, row in top.iterrows():
        rec = row.to_dict()
        rec["record_type"] = "hybrid"
        rec["method"] = ""
        rec["method_display"] = "Hybrid stack"
        rec["point_label"] = f"Hybrid {rec.get('report_stack_id')}"
        rows.append(rec)

        if singular_cache.empty:
            continue
        sub = singular_cache[
            (singular_cache["dataset"].astype(str) == str(row.get("dataset")))
            & (singular_cache["model"].astype(str) == str(row.get("model")))
            & np.isclose(pd.to_numeric(singular_cache["ratio"], errors="coerce"), safe_ratio(row.get("ratio")))
        ].copy()
        if "method" in sub.columns:
            sub = sub.sort_values(["method", "has_checkpoint_path", "timestamp"], ascending=[True, False, False]).drop_duplicates("method", keep="first")
        for _, singular in sub.iterrows():
            srec = singular.to_dict()
            for key in ["objective", "objective_label", "context_rank", "report_stack_id", "stack_id", "selected_methods"]:
                srec[f"hybrid_{key}"] = row.get(key)
            srec["record_type"] = "singular"
            srec["method_display"] = method_display(srec.get("method"))
            srec["point_label"] = srec["method_display"]
            rows.append(srec)
    return pd.DataFrame(rows)


def build_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows = []
    hybrids = metrics[metrics["record_type"].astype(str).eq("hybrid")].copy()
    singulars = metrics[metrics["record_type"].astype(str).eq("singular")].copy()
    for _, h in hybrids.iterrows():
        sub = singulars[
            (singulars["hybrid_report_stack_id"].astype(str) == str(h.get("report_stack_id")))
            & (singulars["dataset"].astype(str) == str(h.get("dataset")))
            & (singulars["model"].astype(str) == str(h.get("model")))
            & np.isclose(pd.to_numeric(singulars["ratio"], errors="coerce"), safe_ratio(h.get("ratio")))
        ]
        for _, s in sub.iterrows():
            for metric, higher_is_better in [
                ("direct_flops_reduction_pct", True),
                ("direct_model_gops", False),
                ("direct_removed_gops", True),
                ("direct_params_reduction_pct", True),
                ("direct_model_params_m", False),
                ("direct_removed_params_m", True),
                ("time_sec", False),
                ("accuracy_pct", True),
                ("accuracy_delta_pp", True),
            ]:
                hv = safe_float(h.get(metric))
                sv = safe_float(s.get(metric))
                advantage = hv - sv if higher_is_better else sv - hv
                rows.append(
                    {
                        "objective": h.get("objective"),
                        "objective_label": h.get("objective_label"),
                        "dataset": h.get("dataset"),
                        "model": h.get("model"),
                        "scope": h.get("scope"),
                        "ratio": h.get("ratio"),
                        "context_rank": h.get("context_rank"),
                        "report_stack_id": h.get("report_stack_id"),
                        "stack_id": h.get("stack_id"),
                        "selected_methods": h.get("selected_methods"),
                        "singular_method": s.get("method"),
                        "singular_method_display": s.get("method_display"),
                        "metric": metric,
                        "metric_source": "checkpoint_direct" if metric.startswith("direct_") else "artifact_index",
                        "hybrid_value": hv,
                        "singular_value": sv,
                        "hybrid_advantage_vs_singular": advantage if math.isfinite(advantage) else math.nan,
                        "hybrid_baseline_gops": safe_float(h.get("direct_baseline_gops")),
                        "hybrid_model_gops": safe_float(h.get("direct_model_gops")),
                        "hybrid_removed_gops": safe_float(h.get("direct_removed_gops")),
                        "singular_baseline_gops": safe_float(s.get("direct_baseline_gops")),
                        "singular_model_gops": safe_float(s.get("direct_model_gops")),
                        "singular_removed_gops": safe_float(s.get("direct_removed_gops")),
                        "hybrid_baseline_params_m": safe_float(h.get("direct_baseline_params_m")),
                        "hybrid_model_params_m": safe_float(h.get("direct_model_params_m")),
                        "hybrid_removed_params_m": safe_float(h.get("direct_removed_params_m")),
                        "singular_baseline_params_m": safe_float(s.get("direct_baseline_params_m")),
                        "singular_model_params_m": safe_float(s.get("direct_model_params_m")),
                        "singular_removed_params_m": safe_float(s.get("direct_removed_params_m")),
                        "hybrid_absolute_metric_provenance": h.get("absolute_metric_provenance"),
                        "singular_absolute_metric_provenance": s.get("absolute_metric_provenance"),
                        "operation_count_convention": h.get("operation_count_convention"),
                    }
                )
    return pd.DataFrame(rows)


def plot_comparison(stack: pd.Series, comparison: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    import matplotlib.pyplot as plt

    sub = comparison[
        (comparison["report_stack_id"].astype(str) == str(stack.get("report_stack_id")))
        & (comparison["objective"].astype(str) == str(stack.get("objective")))
        & (comparison["dataset"].astype(str) == str(stack.get("dataset")))
        & (comparison["model"].astype(str) == str(stack.get("model")))
        & (comparison["scope"].astype(str) == str(stack.get("scope")))
        & np.isclose(pd.to_numeric(comparison["ratio"], errors="coerce"), safe_ratio(stack.get("ratio")))
    ].copy()
    if sub.empty:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    panels = [
        ("accuracy_delta_pp", "Accuracy delta vs baseline (pp)", "#10B981", False, "Hybrid accuracy delta"),
        ("direct_flops_reduction_pct", "Direct structural FLOPs reduction (%)", "#2563EB", False, "Hybrid direct FLOPs"),
        ("time_sec", "Pruning time (s)", "#F97316", True, "Hybrid pruning time"),
    ]
    for ax, (metric, ylabel, color, lower_is_better, hlabel) in zip(axes, panels):
        m = sub[sub["metric"].eq(metric)].dropna(subset=["singular_value"]).copy()
        hv = safe_float(m["hybrid_value"].dropna().iloc[0] if not m["hybrid_value"].dropna().empty else math.nan)
        m = m.sort_values("singular_value", ascending=lower_is_better)
        if not m.empty:
            bars = ax.bar(m["singular_method_display"], m["singular_value"], color=color, alpha=0.78, label="Singular method")
            try:
                if metric == "direct_flops_reduction_pct":
                    labels = [
                        f"{value:.1f}%\n{safe_float(gops):.3f}"
                        if math.isfinite(safe_float(gops))
                        else f"{value:.2f}%"
                        for value, gops in zip(m["singular_value"], m["singular_model_gops"])
                    ]
                else:
                    labels = [f"{v:.2f}" for v in m["singular_value"]]
                ax.bar_label(bars, labels=labels, padding=2, fontsize=7)
            except Exception:
                pass
        if math.isfinite(hv):
            label = f"{hlabel} = {hv:.2f}"
            if metric == "direct_flops_reduction_pct":
                hybrid_gops = safe_float(m["hybrid_model_gops"].dropna().iloc[0]) if not m["hybrid_model_gops"].dropna().empty else math.nan
                if math.isfinite(hybrid_gops):
                    label += f"% / {hybrid_gops:.3f} GOp remaining"
                baseline_gops = safe_float(m["hybrid_baseline_gops"].dropna().iloc[0]) if not m["hybrid_baseline_gops"].dropna().empty else math.nan
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
            ax.axhline(hv, color="#111827", linewidth=1.7, linestyle="--", label=label)
        if metric == "accuracy_delta_pp":
            ax.axhline(0, color="#64748B", linewidth=0.8)
        if metric == "direct_flops_reduction_pct":
            values = pd.to_numeric(m.get("singular_value", pd.Series(dtype=float)), errors="coerce").dropna()
            ymax = max([safe_float(hv, 0.0), *values.tolist(), 1.0])
            ax.set_ylim(top=ymax * 1.18)
            if "singular_model_gops" in m.columns and m["singular_model_gops"].notna().any():
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
        if metric == "direct_flops_reduction_pct":
            ax.set_title("FLOPs reduction and remaining compute", pad=46)
        else:
            ax.set_title(metric.replace("_", " "))
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=55)
        ax.grid(axis="y", alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            if metric == "direct_flops_reduction_pct":
                ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=6.5)
            else:
                ax.legend(handles, labels, fontsize=8)
    fig.suptitle(
        f"V4 checkpoint-derived comparison | {stack.get('report_stack_id')} | {stack.get('objective_label')} | "
        f"{stack.get('dataset')} | {stack.get('model')} | {stack.get('scope')} | r={safe_ratio(stack.get('ratio')):g}",
        fontsize=12,
    )
    fig.tight_layout()
    path = out_dir / (
        f"v4_comparison_{slug(stack.get('objective'))}_{slug(stack.get('dataset'))}_{slug(stack.get('model'))}_"
        f"{slug(stack.get('scope'))}_r{safe_ratio(stack.get('ratio')):g}_rank{int(safe_float(stack.get('context_rank'), 0))}_{slug(stack.get('report_stack_id'))}.png"
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_absolute_footprint(stack: pd.Series, comparison: pd.DataFrame, out_dir: Path) -> Optional[Path]:
    """Plot absolute remaining compute and parameters for appendix use."""
    import matplotlib.pyplot as plt

    sub = comparison[
        (comparison["report_stack_id"].astype(str) == str(stack.get("report_stack_id")))
        & (comparison["objective"].astype(str) == str(stack.get("objective")))
        & (comparison["dataset"].astype(str) == str(stack.get("dataset")))
        & (comparison["model"].astype(str) == str(stack.get("model")))
        & (comparison["scope"].astype(str) == str(stack.get("scope")))
        & np.isclose(pd.to_numeric(comparison["ratio"], errors="coerce"), safe_ratio(stack.get("ratio")))
    ].copy()
    if sub.empty:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    panels = [
        ("direct_model_gops", "Remaining compute (GOp)", "#2563EB", "Hybrid remaining compute"),
        ("direct_model_params_m", "Remaining parameters (millions)", "#7C3AED", "Hybrid remaining parameters"),
    ]
    for ax, (metric, ylabel, color, hybrid_label) in zip(axes, panels):
        metric_rows = sub[sub["metric"].eq(metric)].dropna(subset=["singular_value"]).copy()
        metric_rows = metric_rows.sort_values("singular_value")
        hv = safe_float(metric_rows["hybrid_value"].dropna().iloc[0]) if not metric_rows["hybrid_value"].dropna().empty else math.nan
        if metric_rows.empty:
            ax.text(0.5, 0.5, "No checkpoint-derived values", transform=ax.transAxes, ha="center", va="center")
        else:
            bars = ax.bar(
                metric_rows["singular_method_display"],
                metric_rows["singular_value"],
                color=color,
                alpha=0.72,
                label="Singular method",
            )
            ax.bar_label(bars, labels=[f"{v:.3f}" for v in metric_rows["singular_value"]], padding=2, fontsize=7)
        if math.isfinite(hv):
            ax.axhline(hv, color="#111827", linewidth=1.7, linestyle="--", label=f"{hybrid_label}: {hv:.3f}")
        baseline_col = "hybrid_baseline_gops" if metric == "direct_model_gops" else "hybrid_baseline_params_m"
        baseline = safe_float(metric_rows[baseline_col].dropna().iloc[0]) if not metric_rows.empty and not metric_rows[baseline_col].dropna().empty else math.nan
        if math.isfinite(baseline):
            ax.axhline(baseline, color="#64748B", linewidth=1.2, linestyle=":", label=f"Unpruned baseline: {baseline:.3f}")
        ax.set_title(ylabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=55)
        ax.grid(axis="y", alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=8)
    fig.suptitle(
        f"Absolute checkpoint-derived footprint | {stack.get('report_stack_id')} | {stack.get('objective_label')} | "
        f"{stack.get('dataset')} | {stack.get('model')} | {stack.get('scope')} | r={safe_ratio(stack.get('ratio')):g}",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.01,
        "GOp convention: one multiply-accumulate is counted as one operation.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    path = out_dir / (
        f"v4_absolute_{slug(stack.get('objective'))}_{slug(stack.get('dataset'))}_{slug(stack.get('model'))}_"
        f"{slug(stack.get('scope'))}_r{safe_ratio(stack.get('ratio')):g}_rank{int(safe_float(stack.get('context_rank'), 0))}_{slug(stack.get('report_stack_id'))}.png"
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--base-report-dir", type=Path, default=Path("report_artifacts/context_safe_hybrid_singular_report"))
    parser.add_argument("--registry-dir", type=Path, default=Path("reports/experiment_registry"))
    parser.add_argument("--report-dir", type=Path, default=Path("report_artifacts/context_safe_hybrid_singular_report_v4_model_metrics"))
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument(
        "--force-reprofile",
        action="store_true",
        help="Re-open every checkpoint even when a direct-metrics table already exists.",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    base_report_dir = (project_root / args.base_report_dir).resolve() if not args.base_report_dir.is_absolute() else args.base_report_dir
    registry_dir = (project_root / args.registry_dir).resolve() if not args.registry_dir.is_absolute() else args.registry_dir
    report_dir = (project_root / args.report_dir).resolve() if not args.report_dir.is_absolute() else args.report_dir
    table_dir = report_dir / "tables"
    plot_dir = report_dir / "plots" / "comparisons"
    absolute_plot_dir = report_dir / "plots" / "absolute_footprints"
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    absolute_plot_dir.mkdir(parents=True, exist_ok=True)

    top = read_csv_safe(base_report_dir / "tables" / "top_hybrid_stacks_by_context.csv")
    singular_cache = read_csv_safe(registry_dir / "singular_cache_index.csv")
    if top.empty:
        raise RuntimeError(f"Missing top stack table: {base_report_dir / 'tables' / 'top_hybrid_stacks_by_context.csv'}")
    top["context_rank"] = pd.to_numeric(top["context_rank"], errors="coerce")
    top = top[top["context_rank"].le(args.top_k)].copy()

    checkpoint_index = build_checkpoint_index(top, singular_cache)
    checkpoint_index.to_csv(table_dir / "v4_checkpoint_model_index.csv", index=False)
    if args.max_checkpoints:
        checkpoint_index = checkpoint_index.head(int(args.max_checkpoints)).copy()

    metrics_path = table_dir / "v4_checkpoint_direct_model_metrics.csv"
    cached_metrics = read_csv_safe(metrics_path)
    baseline_cache: Dict[BaselineKey, Tuple[float, float]] = {}
    cached_by_key = {}
    if not args.force_reprofile and not cached_metrics.empty:
        cached_by_key = {
            metric_cache_key(row): row.to_dict()
            for _, row in cached_metrics.iterrows()
        }
    metric_rows = []
    reused_rows = 0
    profiled_rows = 0
    for _, row in checkpoint_index.iterrows():
        key = metric_cache_key(row)
        if key in cached_by_key:
            cached = dict(cached_by_key[key])
            cached.update(row.to_dict())
            metric_rows.append(cached)
            reused_rows += 1
        else:
            metric_rows.append(checkpoint_metrics(row, project_root, args.device, baseline_cache))
            profiled_rows += 1
    metrics = pd.DataFrame(metric_rows)
    print(f"Checkpoint metrics: reused={reused_rows}, profiled={profiled_rows}, total={len(metrics)}")
    metrics = add_absolute_metric_units(metrics)
    metrics.to_csv(table_dir / "v4_checkpoint_direct_model_metrics.csv", index=False)
    baseline_scale = build_baseline_scale_table(metrics)
    baseline_scale.to_csv(table_dir / "v4_baseline_model_scale.csv", index=False)

    comparison = build_comparisons(metrics)
    comparison.to_csv(table_dir / "v4_hybrid_vs_singular_checkpoint_direct_long.csv", index=False)

    layerwise = read_csv_safe(base_report_dir / "tables" / "hybrid_layerwise_policy_linked_to_metrics.csv")
    layerwise_direct = pd.DataFrame()
    if not layerwise.empty:
        hybrid_metric_cols = [
            "objective",
            "dataset",
            "model",
            "scope",
            "ratio",
            "context_rank",
            "report_stack_id",
            "stack_id",
            "selected_methods",
            "metric_status",
            "metric_error",
            "checkpoint_path_resolved",
            "direct_baseline_flops",
            "direct_model_flops",
            "direct_flops_reduction_pct",
            "direct_baseline_params",
            "direct_model_params",
            "direct_params_reduction_pct",
        ]
        hybrid_direct = metrics[
            metrics.get("record_type", pd.Series(dtype=str)).astype(str).eq("hybrid")
        ][[c for c in hybrid_metric_cols if c in metrics.columns]].copy()
        join_cols = [
            c
            for c in ["objective", "dataset", "model", "scope", "ratio", "context_rank", "report_stack_id"]
            if c in layerwise.columns and c in hybrid_direct.columns
        ]
        if join_cols:
            layerwise_direct = layerwise.merge(
                hybrid_direct,
                on=join_cols,
                how="left",
                suffixes=("", "_checkpoint_direct"),
            )
        else:
            layerwise_direct = layerwise.copy()
    layerwise_direct.to_csv(table_dir / "v4_hybrid_layerwise_policy_linked_to_direct_metrics.csv", index=False)

    plot_rows = []
    hybrids = metrics[metrics.get("record_type", pd.Series(dtype=str)).astype(str).eq("hybrid")].copy()
    for _, stack in hybrids.iterrows():
        path = plot_comparison(stack, comparison, plot_dir)
        absolute_path = plot_absolute_footprint(stack, comparison, absolute_plot_dir)
        plot_rows.append(
            {
                **{k: stack.get(k) for k in CONTEXT_KEYS},
                "objective_label": stack.get("objective_label"),
                "context_rank": stack.get("context_rank"),
                "report_stack_id": stack.get("report_stack_id"),
                "stack_id": stack.get("stack_id"),
                "comparison_plot": str(path) if path else "",
                "absolute_footprint_plot": str(absolute_path) if absolute_path else "",
            }
        )
    plot_manifest = pd.DataFrame(plot_rows)
    plot_manifest.to_csv(table_dir / "v4_plot_manifest_checkpoint_comparisons.csv", index=False)

    qc_rows = [
        {"metric": "checkpoint_index_rows", "value": int(len(checkpoint_index))},
        {"metric": "metrics_rows", "value": int(len(metrics))},
        {"metric": "metrics_ok_rows", "value": int(metrics.get("metric_status", pd.Series(dtype=str)).astype(str).eq("ok").sum()) if not metrics.empty else 0},
        {"metric": "comparison_rows", "value": int(len(comparison))},
        {"metric": "comparison_plots", "value": int(plot_manifest.get("comparison_plot", pd.Series(dtype=str)).astype(str).ne("").sum()) if not plot_manifest.empty else 0},
        {"metric": "absolute_footprint_plots", "value": int(plot_manifest.get("absolute_footprint_plot", pd.Series(dtype=str)).astype(str).ne("").sum()) if not plot_manifest.empty else 0},
        {"metric": "baseline_scale_rows", "value": int(len(baseline_scale))},
        {"metric": "layerwise_direct_rows", "value": int(len(layerwise_direct))},
    ]
    qc = pd.DataFrame(qc_rows)
    qc.to_csv(table_dir / "v4_qc_summary.csv", index=False)
    manifest = {
        "report_dir": str(report_dir),
        "tables_dir": str(table_dir),
        "plots_dir": str(plot_dir),
        "top_k": args.top_k,
        "device": args.device,
        "counts": {r["metric"]: r["value"] for r in qc_rows},
        "notes": [
            "FLOPs and params are recomputed from checkpoint-loaded model objects.",
            "Absolute compute is reported in GOp with one multiply-accumulate counted as one operation.",
            "Accuracy and pruning time remain indexed from benchmark artifacts unless a future eval pass is enabled.",
            "CSV artifacts are used only for context/linkage, not for direct structural FLOPs.",
        ],
    }
    (report_dir / "v4_report_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
