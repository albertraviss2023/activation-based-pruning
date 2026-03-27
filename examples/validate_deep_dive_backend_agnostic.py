from __future__ import annotations

import gc
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from reducnn.backends.factory import get_adapter
from reducnn.pruner.mask_builder import build_pruning_masks
from reducnn.visualization.animator import PruningAnimator


@dataclass
class BackendValidationResult:
    backend: str
    adapter: str
    input_shape: tuple
    load_attempts: Dict[str, str]
    runtime_model: str
    score_layers: int
    trace_nodes: int
    trace_clusters: int


def _model_candidates() -> List[str]:
    return ["resnet18", "vgg16", "densenet121", "mobilenet_v2"]


def _make_runtime_loader(backend: str, batch_size: int = 4, num_classes: int = 10):
    if backend == "pytorch":
        import torch

        x = torch.randn(batch_size, 3, 32, 32)
        y = torch.randint(0, num_classes, (batch_size,))
        return [(x, y)]

    x = np.random.randn(batch_size, 32, 32, 3).astype("float32")
    y = np.random.randint(0, num_classes, size=(batch_size,), dtype="int32")
    return [(x, y)]


def _validate_backend(backend: str) -> BackendValidationResult:
    cfg = {
        "backend": backend,
        "model_type": "resnet18",
        "input_shape": (3, 32, 32) if backend == "pytorch" else (32, 32, 3),
        "num_classes": 10,
        "keras_weights": "none",
        "experiment_id": f"deep_dive_{backend}_agnostic_check",
    }
    adapter = get_adapter(None, cfg)
    load_attempts: Dict[str, str] = {}

    # 1) Prove broad model-loading compatibility for deep-dive candidates.
    for model_name in _model_candidates():
        model = None
        try:
            model = adapter.get_model(
                model_name,
                input_shape=cfg["input_shape"],
                num_classes=cfg["num_classes"],
                pretrained=False,
            )
            load_attempts[model_name] = "ok"
        except Exception as e:  # pragma: no cover - report includes errors for debugging
            load_attempts[model_name] = f"error: {type(e).__name__}: {e}"
        finally:
            del model
            gc.collect()

    # 2) Prove visualization backend-path works end-to-end on a runtime model.
    runtime_model = adapter.get_model(
        "vgg16",
        input_shape=cfg["input_shape"],
        num_classes=cfg["num_classes"],
        pretrained=False,
    )
    loader = _make_runtime_loader(backend)
    score_map = adapter.get_score_map(runtime_model, loader, "apoz")
    graph = adapter.trace_graph(runtime_model)
    masks = build_pruning_masks(
        score_map,
        ratio=0.2,
        scope="local",
        clusters=graph.get("clusters", {}),
    )
    animator = PruningAnimator(adapter)
    trace = animator.build_pruning_trace(
        model=runtime_model,
        score_map=score_map,
        masks=masks,
        method_name="apoz",
        candidate_ratio=0.2,
    )

    return BackendValidationResult(
        backend=backend,
        adapter=type(adapter).__name__,
        input_shape=tuple(cfg["input_shape"]),
        load_attempts=load_attempts,
        runtime_model="vgg16",
        score_layers=len(score_map),
        trace_nodes=int(trace["meta"]["node_count"]),
        trace_clusters=int(trace["meta"]["cluster_count"]),
    )


def main():
    results = []

    # PyTorch path
    results.append(_validate_backend("pytorch"))

    # Keras path (skip gracefully if tensorflow is unavailable)
    try:
        import tensorflow as _  # noqa: F401

        results.append(_validate_backend("keras"))
    except Exception as e:
        results.append(
            BackendValidationResult(
                backend="keras",
                adapter="unavailable",
                input_shape=(32, 32, 3),
                load_attempts={"keras_backend": f"skipped: {type(e).__name__}: {e}"},
                runtime_model="n/a",
                score_layers=0,
                trace_nodes=0,
                trace_clusters=0,
            )
        )

    out_dir = Path("outputs") / "viz_deep_dive" / "agnostic_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "backend_agnostic_report.json"
    payload = {"results": [asdict(r) for r in results]}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved backend-agnostic report: {out_path}")
    for r in results:
        ok_count = sum(1 for v in r.load_attempts.values() if str(v) == "ok")
        total = len(r.load_attempts)
        print(
            f"[{r.backend}] adapter={r.adapter} loaded={ok_count}/{total} "
            f"score_layers={r.score_layers} trace_nodes={r.trace_nodes}"
        )


if __name__ == "__main__":
    main()
