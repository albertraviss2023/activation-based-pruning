from __future__ import annotations

import numpy as np
import pytest

from reducnn.backends.factory import get_adapter
from reducnn.pruner.mask_builder import build_pruning_masks
from reducnn.visualization.animator import PruningAnimator


def _torch_loader(batch_size: int = 2, num_classes: int = 10):
    torch = pytest.importorskip("torch")
    x = torch.randn(batch_size, 3, 32, 32)
    y = torch.randint(0, num_classes, (batch_size,))
    return [(x, y)]


def _keras_loader(batch_size: int = 2, num_classes: int = 10):
    np.random.seed(7)
    x = np.random.randn(batch_size, 32, 32, 3).astype("float32")
    y = np.random.randint(0, num_classes, size=(batch_size,), dtype="int32")
    return [(x, y)]


def _run_backend_smoke(backend: str):
    cfg = {
        "backend": backend,
        "model_type": "resnet18",
        "input_shape": (3, 32, 32) if backend == "pytorch" else (32, 32, 3),
        "num_classes": 10,
        "keras_weights": "none",
        "experiment_id": f"test_deep_dive_{backend}",
    }
    adapter = get_adapter(None, cfg)

    # Requested resnet18 should work on both backends (Keras uses aliasing).
    model = adapter.get_model(
        "resnet18",
        input_shape=cfg["input_shape"],
        num_classes=cfg["num_classes"],
        pretrained=False,
    )
    assert model is not None

    # Deep-dive visualization backend path smoke check.
    viz_model = adapter.get_model(
        "vgg16",
        input_shape=cfg["input_shape"],
        num_classes=cfg["num_classes"],
        pretrained=False,
    )
    loader = _torch_loader() if backend == "pytorch" else _keras_loader()
    score_map = adapter.get_score_map(viz_model, loader, "apoz")
    assert score_map

    graph = adapter.trace_graph(viz_model)
    masks = build_pruning_masks(score_map, ratio=0.2, scope="local", clusters=graph.get("clusters", {}))
    animator = PruningAnimator(adapter)
    trace = animator.build_pruning_trace(
        model=viz_model,
        score_map=score_map,
        masks=masks,
        method_name="apoz",
        candidate_ratio=0.2,
    )
    assert trace["meta"]["node_count"] > 0


def test_deep_dive_backend_agnostic_pytorch():
    _run_backend_smoke("pytorch")


def test_deep_dive_backend_agnostic_keras():
    pytest.importorskip("tensorflow")
    _run_backend_smoke("keras")
