"""Run a tiny CIFAR-10 method-matrix and adaptive-hybrid smoke test.

This is intentionally small and CPU-friendly. It is meant to validate that all
registered methods can be scored/pruned, that efficiency JSONs are emitted, and
that the adaptive hybrid can consume those JSONs. It is not a thesis-quality
accuracy experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from reducnn.analyzer.classifier import ArchitectureClassifier
from reducnn.backends.factory import get_adapter
from reducnn.pruner import ReduCNNPruner
from reducnn.pruner.mask_builder import build_pruning_masks
from reducnn.pruner.meta_criteria import HybridMetaPruner
from ui.pruning_methods import register_ui_methods


MODELS = ["resnet18", "vgg16", "densenet121", "mobilenet_v2"]
SIMPLE_METHODS = ["l1_norm", "custom_l2", "mean_abs_act", "apoz"]
MEDIUM_METHODS = [
    "custom_entropy",
    "custom_class_entropy",
    "custom_hrank",
    "custom_spectral_energy",
    "chip",
    "custom_reprune",
]
COMPLEX_METHODS = ["custom_tis", "custom_nisp", "custom_thinet", "custom_senpis"]
ALL_METHODS = SIMPLE_METHODS + MEDIUM_METHODS + COMPLEX_METHODS
GLOBAL_FORMULATION_METHODS = {"chip", "custom_nisp", "custom_senpis", "custom_tis", "custom_reprune", "custom_thinet"}
THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def make_loader(train: bool, samples: int, batch_size: int, shuffle: bool = False) -> DataLoader:
    tfm = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    ds = torchvision.datasets.CIFAR10(root=str(ROOT / "data"), train=train, download=True, transform=tfm)
    idx = list(range(min(samples, len(ds))))
    return DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def round_or_none(v: Any, digits: int = 6) -> Any:
    try:
        if v is None:
            return None
        f = float(v)
        if not np.isfinite(f):
            return None
        return round(f, digits)
    except Exception:
        return None


def pct_reduction(baseline: Any, current: Any) -> Any:
    try:
        b = float(baseline)
        c = float(current)
        if b <= 0 or not np.isfinite(b) or not np.isfinite(c):
            return None
        return 100.0 * (b - c) / b
    except Exception:
        return None


def safe_eval(adapter: Any, model: Any, loader: DataLoader) -> Any:
    try:
        return float(adapter.evaluate(model, loader))
    except Exception:
        return None


def safe_stats(adapter: Any, model: Any, loader: DataLoader) -> Tuple[Any, Any]:
    try:
        flops, params = adapter.get_stats(model, loader)
        return float(flops), float(params)
    except Exception:
        return None, None


def mask_stats(masks: Dict[str, np.ndarray]) -> Tuple[int, float, float]:
    keeps = []
    for m in masks.values():
        arr = np.asarray(m).reshape(-1)
        if arr.size:
            keeps.append(float(np.mean(arr.astype(bool))))
    if not keeps:
        return 0, 0.0, 0.0
    return len(keeps), float(np.mean(keeps)), float(np.min(keeps))


def metrics_payload(prefix: str, acc: Any, flops: Any, params: Any, b_acc: Any, b_flops: Any, b_params: Any) -> Dict[str, Any]:
    return {
        f"{prefix}_accuracy_pct": round_or_none(acc, 6),
        f"{prefix}_accuracy_delta_pct": round_or_none(None if acc is None or b_acc is None else float(acc) - float(b_acc), 6),
        f"{prefix}_flops": round_or_none(flops, 3),
        f"{prefix}_flops_reduction_pct": round_or_none(pct_reduction(b_flops, flops), 6),
        f"{prefix}_params": round_or_none(params, 3),
        f"{prefix}_params_reduction_pct": round_or_none(pct_reduction(b_params, params), 6),
    }


def write_records(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        csv_path = path.with_suffix(".csv")
        fields = sorted({k for r in rows for k in r})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def rank_efficiency(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ok = [dict(r) for r in rows if str(r.get("status", "")).lower() == "ok"]

    def key(r: Dict[str, Any]) -> Tuple[float, float, float, float]:
        return (
            float(r.get("simplicity_time_sec") or 1e12),
            -float(r.get("healed_accuracy_delta_pct") or -1e12),
            -float(r.get("healed_flops_reduction_pct") or -1e12),
            -float(r.get("healed_params_reduction_pct") or -1e12),
        )

    ranked = sorted(ok, key=key)
    for i, r in enumerate(ranked, start=1):
        r["efficiency_rank"] = i
        r["efficiency_rank_basis"] = "simplicity_time_sec, healed_accuracy_delta_pct, healed_flops_reduction_pct, healed_params_reduction_pct"
        r["efficiency_accuracy_delta_pct"] = r.get("healed_accuracy_delta_pct")
        r["efficiency_flops_reduction_pct"] = r.get("healed_flops_reduction_pct")
        r["efficiency_params_reduction_pct"] = r.get("healed_params_reduction_pct")
    return ranked


def make_adapter(model_name: str, batch_size: int, calib_batches: int) -> Any:
    return get_adapter(
        None,
        {
            "backend": "pytorch",
            "model_type": model_name,
            "input_shape": (3, 32, 32),
            "num_classes": 10,
            "prune_batches": calib_batches,
            "chip_max_spatial": 8,
            "torch_amp": False,
            "torch_restore_best": False,
            "torch_reduce_lr_on_plateau": False,
            "lr": 1e-4,
            "batch_size": batch_size,
            "dataset": "cifar-10",
        },
    )


def fresh_model(adapter: Any, model_name: str) -> Any:
    model = adapter.get_model(model_name, input_shape=(3, 32, 32), num_classes=10, pretrained=False)
    model.eval()
    return model


def run_method_matrix(model_name: str, args: argparse.Namespace, out_root: Path) -> Tuple[Path, List[Dict[str, Any]]]:
    print(f"\n=== Method matrix smoke: {model_name} ===", flush=True)
    adapter = make_adapter(model_name, args.batch_size, args.calib_batches)
    calib_loader = make_loader(train=True, samples=args.calib_samples, batch_size=args.batch_size)
    eval_loader = make_loader(train=False, samples=args.eval_samples, batch_size=args.batch_size)

    baseline = fresh_model(adapter, model_name)
    baseline_acc = safe_eval(adapter, baseline, eval_loader)
    baseline_flops, baseline_params = safe_stats(adapter, baseline, eval_loader)
    print(f"[{model_name}] baseline acc={baseline_acc} flops={baseline_flops} params={baseline_params}", flush=True)

    rows: List[Dict[str, Any]] = []
    for method in ALL_METHODS:
        scope = "global" if method in GLOBAL_FORMULATION_METHODS else "local"
        print(f"[{model_name}] method={method} scope={scope}", flush=True)
        t0 = time.time()
        status = "ok"
        err = ""
        n_layers = 0
        mean_keep = 0.0
        min_keep = 0.0
        raw_acc = raw_flops = raw_params = None
        healed_acc = healed_flops = healed_params = None
        prune_time = 0.0
        heal_time = 0.0
        try:
            model = fresh_model(adapter, model_name)
            pruner = ReduCNNPruner(
                method=method,
                scope=scope,
                config={"backend": "pytorch", "chip_max_spatial": 8, "ratio": args.prune_ratio},
            )
            prune_start = time.time()
            pruned_model, masks, _ = pruner.prune_custom_model(model, calib_loader, ratio=args.prune_ratio)
            prune_time = time.time() - prune_start
            n_layers, mean_keep, min_keep = mask_stats(masks)
            raw_acc = safe_eval(adapter, pruned_model, eval_loader)
            raw_flops, raw_params = safe_stats(adapter, pruned_model, eval_loader)

            heal_start = time.time()
            if args.heal_epochs > 0:
                train_loader = make_loader(train=True, samples=args.heal_samples, batch_size=args.batch_size, shuffle=True)
                adapter.train(pruned_model, train_loader, args.heal_epochs, name=f"smoke_{model_name}_{method}_heal", plot=False)
            heal_time = time.time() - heal_start
            healed_acc = safe_eval(adapter, pruned_model, eval_loader)
            healed_flops, healed_params = safe_stats(adapter, pruned_model, eval_loader)
        except Exception as e:
            status = "error"
            err = f"{type(e).__name__}: {e}"
            print(f"[{model_name}] {method} failed: {err}", flush=True)

        row = {
            "backend": "pytorch",
            "dataset": "cifar-10",
            "model": model_name,
            "method": method,
            "scope": scope,
            "prune_ratio": args.prune_ratio,
            "status": status,
            "error": err,
            "layers_scored": n_layers,
            "mean_keep_ratio": round_or_none(mean_keep, 6),
            "min_keep_ratio": round_or_none(min_keep, 6),
            "baseline_accuracy_pct": round_or_none(baseline_acc, 6),
            "baseline_flops": round_or_none(baseline_flops, 3),
            "baseline_params": round_or_none(baseline_params, 3),
            "prune_time_sec": round_or_none(prune_time, 6),
            "heal_time_sec": round_or_none(heal_time, 6),
            "simplicity_time_sec": round_or_none(float(prune_time or 0.0) + float(heal_time or 0.0), 6),
            "wall_time_sec": round_or_none(time.time() - t0, 6),
            "smoke_test": True,
        }
        row.update(metrics_payload("raw_pruned", raw_acc, raw_flops, raw_params, baseline_acc, baseline_flops, baseline_params))
        row.update(metrics_payload("healed", healed_acc, healed_flops, healed_params, baseline_acc, baseline_flops, baseline_params))
        rows.append(row)

    stamp = now_stamp()
    base = out_root / f"smoke_cifar10_{model_name}_{stamp}"
    write_records(rows, base.with_suffix(".json"))
    efficiency = rank_efficiency(rows)
    efficiency_path = out_root / f"smoke_cifar10_{model_name}_{stamp}_efficiency.json"
    write_records(efficiency, efficiency_path)
    print(f"[{model_name}] efficiency JSON: {efficiency_path}", flush=True)
    return efficiency_path, rows


def decision_to_row(model_name: str, threshold: float, ratio: float, layer: str, decision: Dict[str, Any]) -> Dict[str, Any]:
    stack = decision.get("stack_methods") or list((decision.get("weights") or {}).keys())
    covered = decision.get("covered_complex_methods") or []
    return {
        "dataset": "cifar-10",
        "model": model_name,
        "prune_ratio": ratio,
        "correlation_threshold": threshold,
        "layer": layer,
        "mode": decision.get("mode", ""),
        "selected": decision.get("selected", ""),
        "stack_methods": "|".join(stack),
        "stack_size": len(stack),
        "covered_complex_methods": "|".join(covered),
        "weights_json": json.dumps(decision.get("weights", {}), sort_keys=True),
        "similarity_rule": decision.get("similarity_rule", ""),
    }


def run_hybrid(model_name: str, efficiency_path: Path, args: argparse.Namespace, out_root: Path) -> Dict[str, Any]:
    print(f"\n=== Adaptive hybrid smoke: {model_name} ===", flush=True)
    adapter = make_adapter(model_name, args.batch_size, args.calib_batches)
    adapter.config.update(
        {
            "dataset": "cifar-10",
            "hybrid_efficiency_json_path": str(efficiency_path),
            "hybrid_metric_pool": ALL_METHODS,
            "hybrid_simple_methods": SIMPLE_METHODS,
            "hybrid_allow_simple_only_stack": True,
            "hybrid_simple_proxy_only": True,
            "hybrid_include_best_simple_representative": True,
            "hybrid_topk_overlap_threshold": args.overlap_threshold,
            "current_prune_ratio": args.prune_ratio,
            "ratio": args.prune_ratio,
        }
    )
    calib_loader = make_loader(train=True, samples=args.calib_samples, batch_size=args.batch_size)
    eval_loader = make_loader(train=False, samples=args.eval_samples, batch_size=args.batch_size)
    model = fresh_model(adapter, model_name)
    baseline_acc = safe_eval(adapter, model, eval_loader)
    baseline_flops, baseline_params = safe_stats(adapter, model, eval_loader)

    graph = adapter.trace_graph(model)
    layers = [n for n, d in graph["nodes"].items() if d.get("type") == "conv2d"]
    engine = HybridMetaPruner(adapter, mode="adaptive")

    t0 = time.time()
    multi_scores = adapter.get_multi_metric_scores(model, calib_loader, ALL_METHODS)
    score_time = time.time() - t0

    decision_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    selected_score_map = None
    selected_decisions = None
    for threshold in THRESHOLDS:
        adapter.config["hybrid_correlation_threshold"] = threshold
        score_map = {}
        layer_decisions = {}
        for layer in layers:
            score, decision = engine._adaptive_layer_score(multi_scores, layer)
            score_map[layer] = score
            layer_decisions[layer] = decision
            decision_rows.append(decision_to_row(model_name, threshold, args.prune_ratio, layer, decision))
        modes = [d.get("mode", "") for d in layer_decisions.values()]
        stack_sizes = [len(d.get("stack_methods") or list((d.get("weights") or {}).keys())) for d in layer_decisions.values()]
        summary_rows.append(
            {
                "dataset": "cifar-10",
                "model": model_name,
                "prune_ratio": args.prune_ratio,
                "correlation_threshold": threshold,
                "layers": len(layer_decisions),
                "cheap_proxy_layers": modes.count("cheap_proxy"),
                "blend_layers": modes.count("cost_aware_blend"),
                "simple_only_layers": modes.count("simple_only_single") + modes.count("simple_only_stack"),
                "avg_stack_size": round_or_none(float(np.mean(stack_sizes)) if stack_sizes else 0.0, 6),
                "max_stack_size": max(stack_sizes) if stack_sizes else 0,
                "score_time_sec": round_or_none(score_time, 6),
            }
        )
        if abs(threshold - args.demo_threshold) < 1e-9:
            selected_score_map = score_map
            selected_decisions = layer_decisions

    if selected_score_map is None:
        selected_score_map = score_map
        selected_decisions = layer_decisions

    demo = {
        "dataset": "cifar-10",
        "model": model_name,
        "prune_ratio": args.prune_ratio,
        "correlation_threshold": args.demo_threshold,
        "baseline_accuracy_pct": round_or_none(baseline_acc, 6),
        "baseline_flops": round_or_none(baseline_flops, 3),
        "baseline_params": round_or_none(baseline_params, 3),
        "status": "ok",
        "error": "",
        "score_time_sec": round_or_none(score_time, 6),
    }
    try:
        clusters = ArchitectureClassifier(adapter).get_clusters(model)
        masks = build_pruning_masks(selected_score_map, args.prune_ratio, scope="local", clusters=clusters)
        n_layers, mean_keep, min_keep = mask_stats(masks)
        prune_start = time.time()
        pruned = adapter.apply_surgery(model, masks)
        surgery_time = time.time() - prune_start
        raw_acc = safe_eval(adapter, pruned, eval_loader)
        raw_flops, raw_params = safe_stats(adapter, pruned, eval_loader)
        demo.update(
            {
                "layers_scored": n_layers,
                "mean_keep_ratio": round_or_none(mean_keep, 6),
                "min_keep_ratio": round_or_none(min_keep, 6),
                "surgery_time_sec": round_or_none(surgery_time, 6),
                "simplicity_time_sec": round_or_none(score_time + surgery_time, 6),
            }
        )
        demo.update(metrics_payload("raw_pruned", raw_acc, raw_flops, raw_params, baseline_acc, baseline_flops, baseline_params))
    except Exception as e:
        demo["status"] = "error"
        demo["error"] = f"{type(e).__name__}: {e}"
        print(f"[{model_name}] hybrid demo surgery failed: {demo['error']}", flush=True)

    stamp = now_stamp()
    model_dir = out_root / "adaptive_hybrid" / model_name / stamp
    write_records(decision_rows, model_dir / "layer_decisions.json")
    write_records(summary_rows, model_dir / "threshold_summary.json")
    write_records([demo], model_dir / "adaptive_hybrid_demo_summary.json")
    print(f"[{model_name}] hybrid outputs: {model_dir}", flush=True)
    return {
        "model": model_name,
        "efficiency_path": str(efficiency_path),
        "hybrid_dir": str(model_dir),
        "hybrid_demo": demo,
        "threshold_summary": summary_rows,
        "decision_count": len(decision_rows),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--calib-samples", type=int, default=16)
    p.add_argument("--eval-samples", type=int, default=16)
    p.add_argument("--heal-samples", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--calib-batches", type=int, default=1)
    p.add_argument("--prune-ratio", type=float, default=0.30)
    p.add_argument("--heal-epochs", type=int, default=0)
    p.add_argument("--demo-threshold", type=float, default=0.60)
    p.add_argument("--overlap-threshold", type=float, default=0.80)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    register_ui_methods()
    out_root = ROOT / "outputs" / "smoke_cifar10"
    out_root.mkdir(parents=True, exist_ok=True)
    run_summary = []
    for model_name in args.models:
        efficiency_path, _ = run_method_matrix(model_name, args, out_root / "custom_method_matrix")
        run_summary.append(run_hybrid(model_name, efficiency_path, args, out_root))
    summary_path = out_root / f"smoke_cifar10_all_models_summary_{now_stamp()}.json"
    write_records(run_summary, summary_path)
    print(f"\nALL DONE: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
