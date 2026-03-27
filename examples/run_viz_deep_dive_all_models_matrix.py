from __future__ import annotations

import gc
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from reducnn.backends.factory import get_adapter
from reducnn.pruner.mask_builder import build_pruning_masks
from reducnn.visualization.animator import PruningAnimator


def _candidates_for_backend(backend: str) -> List[str]:
    # Use the same request names used in the deep-dive notebook.
    return ["resnet18", "vgg16", "densenet121", "mobilenet_v2"]


def _make_loader(backend: str, batch_size: int = 2, num_classes: int = 10):
    if backend == "pytorch":
        import torch

        x = torch.randn(batch_size, 3, 32, 32)
        y = torch.randint(0, num_classes, (batch_size,))
        return [(x, y)]
    x = np.random.randn(batch_size, 32, 32, 3).astype("float32")
    y = np.random.randint(0, num_classes, size=(batch_size,), dtype="int32")
    return [(x, y)]


def _clear_backend_state(backend: str):
    if backend == "keras":
        try:
            import tensorflow as tf

            tf.keras.backend.clear_session()
        except Exception:
            pass
    gc.collect()


def _check_single_model(adapter, backend: str, model_name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "requested_model": model_name,
        "status": "error",
        "error": None,
        "elapsed_sec": 0.0,
        "score_layers": 0,
        "trace_nodes": 0,
        "trace_clusters": 0,
    }

    t0 = time.time()
    model = None
    try:
        model = adapter.get_model(
            model_name,
            input_shape=cfg["input_shape"],
            num_classes=cfg["num_classes"],
            pretrained=False,
        )
        loader = _make_loader(backend=backend, batch_size=2, num_classes=cfg["num_classes"])

        # Deep-dive core path: score -> mask -> trace -> phase-3 figs.
        metrics = adapter.get_multi_metric_scores(model, loader, ["apoz", "l1_norm", "mean_abs_act"])
        score_map = metrics["apoz"]
        graph = adapter.trace_graph(model)
        masks = build_pruning_masks(score_map, ratio=0.2, scope="local", clusters=graph.get("clusters", {}))

        animator = PruningAnimator(adapter)
        trace = animator.build_pruning_trace(
            model=model,
            score_map=score_map,
            masks=masks,
            method_name="apoz",
            candidate_ratio=0.2,
        )

        # Smoke render objects for phase-3/phase-4 visuals.
        fig_a = animator.generate_candidate_discovery_graph(
            model=model,
            score_map=score_map,
            masks=masks,
            method_name="apoz",
            candidate_ratio=0.2,
        )
        fig_b = animator.generate_pruning_process_animation(
            model=model,
            score_map=score_map,
            masks=masks,
            method_name="apoz",
            candidate_ratio=0.2,
        )
        fig_c = animator.generate_architecture_comparison(
            model=model,
            masks=masks,
            method_name="apoz",
        )

        out["status"] = "ok"
        out["score_layers"] = int(len(score_map))
        out["trace_nodes"] = int(trace["meta"]["node_count"])
        out["trace_clusters"] = int(trace["meta"]["cluster_count"])
        out["candidate_traces"] = int(len(getattr(fig_a, "data", [])))
        out["process_frames"] = int(len(getattr(fig_b, "frames", [])))
        out["arch_traces"] = int(len(getattr(fig_c, "data", [])))
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    finally:
        out["elapsed_sec"] = round(time.time() - t0, 2)
        del model
        _clear_backend_state(backend)

    return out


def _run_backend(backend: str) -> Dict[str, Any]:
    cfg = {
        "backend": backend,
        "model_type": "resnet18",
        "input_shape": (3, 32, 32) if backend == "pytorch" else (32, 32, 3),
        "num_classes": 10,
        "keras_weights": "none",
        "prune_batches": 1,  # keep smoke test fast
        "experiment_id": f"viz_deep_dive_matrix_{backend}",
    }
    adapter = get_adapter(None, cfg)

    models = _candidates_for_backend(backend)
    rows = [_check_single_model(adapter, backend, m, cfg) for m in models]
    passed = sum(1 for r in rows if r["status"] == "ok")
    return {
        "backend": backend,
        "adapter": type(adapter).__name__,
        "input_shape": list(cfg["input_shape"]),
        "models_tested": models,
        "passed": passed,
        "total": len(rows),
        "rows": rows,
    }


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload: Dict[str, Any] = {
        "generated_at": stamp,
        "scope": "visualization_deep_dive all-model smoke matrix",
        "results": [],
    }

    payload["results"].append(_run_backend("pytorch"))
    try:
        import tensorflow as _  # noqa: F401

        payload["results"].append(_run_backend("keras"))
    except Exception as e:
        payload["results"].append(
            {
                "backend": "keras",
                "adapter": "unavailable",
                "input_shape": [32, 32, 3],
                "models_tested": [],
                "passed": 0,
                "total": 0,
                "rows": [],
                "error": f"{type(e).__name__}: {e}",
            }
        )

    out_dir = Path("outputs") / "viz_deep_dive" / "agnostic_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"all_models_matrix_{stamp}.json"
    latest_path = out_dir / "all_models_matrix_latest.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved matrix report: {out_path}")
    print(f"Updated latest report: {latest_path}")
    for r in payload["results"]:
        print(
            f"[{r['backend']}] adapter={r['adapter']} pass={r.get('passed', 0)}/{r.get('total', 0)}"
        )
        for row in r.get("rows", []):
            state = row["status"]
            model = row["requested_model"]
            dt = row["elapsed_sec"]
            if state == "ok":
                print(
                    f"  - {model}: ok ({dt}s) score_layers={row['score_layers']} trace_nodes={row['trace_nodes']}"
                )
            else:
                print(f"  - {model}: ERROR ({dt}s) {row.get('error')}")


if __name__ == "__main__":
    main()
