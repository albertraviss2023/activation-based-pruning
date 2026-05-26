from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from reducnn.analyzer.classifier import ArchitectureClassifier
from reducnn.backends.torch_backend import PyTorchAdapter
from reducnn.engine.orchestrator import Orchestrator
from reducnn.pruner.chip import chip_channel_independence_scores
from reducnn.pruner.mask_builder import build_pruning_masks
from reducnn.pruner.meta_criteria import HybridMetaPruner
from reducnn.pruner.registry import register_method


def test_apoz_keeps_active_channels_torch():
    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 2, kernel_size=1, bias=False)

        def forward(self, x):
            return self.conv(x)

    model = TinyNet()
    with torch.no_grad():
        model.conv.weight.zero_()
        model.conv.weight[1, 0, 0, 0] = 1.0  # channel 1 active, channel 0 dead

    x = torch.ones(4, 1, 4, 4)
    y = torch.zeros(4, dtype=torch.long)
    loader = [(x, y)]

    adapter = PyTorchAdapter({"input_shape": (1, 4, 4), "num_classes": 2})
    score = adapter.get_score_map(model, loader, "apoz")["conv"]
    mask = build_pruning_masks({"conv": score}, ratio=0.5, scope="local")["conv"]

    assert mask.shape[0] == 2
    assert bool(mask[1]) is True
    assert bool(mask[0]) is False


def test_chip_scores_identical_channels_low_independence():
    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 2, kernel_size=3, padding=1, bias=False)

        def forward(self, x):
            return self.conv(x)

    model = TinyNet()
    with torch.no_grad():
        w = torch.randn(3, 3, 3)
        model.conv.weight[0] = w
        model.conv.weight[1] = w  # identical filters => highly correlated channels

    x = torch.randn(6, 3, 16, 16)
    y = torch.randint(0, 2, (6,))
    loader = [(x, y)]

    @register_method("chip_regression_test", framework="torch")
    def _chip_regression_torch(layer, **kwargs):
        activations = []
        hook = layer.register_forward_hook(lambda _m, _i, o: activations.append((o[0] if isinstance(o, tuple) else o).detach().cpu().numpy()))
        try:
            kwargs["model"].eval()
            with torch.no_grad():
                for xb, _yb in kwargs["loader"]:
                    kwargs["model"](xb)
        finally:
            hook.remove()
        if not activations:
            return None
        act = np.concatenate(activations, axis=0)
        return chip_channel_independence_scores(act, channel_axis=1, max_spatial=64)

    adapter = PyTorchAdapter({"chip_max_spatial": 64})
    scores = adapter.get_score_map(model, loader, "chip_regression_test")["conv"]

    assert np.all(np.isfinite(scores))
    assert np.max(scores) < 0.2  # both channels should have low independence
    assert abs(float(scores[0]) - float(scores[1])) < 1e-4


def test_residual_clusters_are_disjoint_after_merge():
    adapter = PyTorchAdapter({"input_shape": (3, 32, 32), "num_classes": 10})
    model = adapter.get_model("resnet18")
    clusters = ArchitectureClassifier(adapter).get_clusters(model)

    member_sets = [set(v) for _, v in sorted(clusters.items())]
    for i in range(len(member_sets)):
        for j in range(i + 1, len(member_sets)):
            assert member_sets[i].isdisjoint(member_sets[j])


def test_orchestrator_runs_with_explicit_adapter_and_model(monkeypatch):
    # Disable plotting side effects for headless test runs.
    monkeypatch.setattr("reducnn.engine.orchestrator.plot_training_history", lambda *a, **k: None)
    monkeypatch.setattr("reducnn.engine.orchestrator.plot_layer_sensitivity", lambda *a, **k: None)
    monkeypatch.setattr("reducnn.engine.orchestrator.plot_metrics_comparison", lambda *a, **k: None)

    class TinyClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Linear(8, 10)

        def forward(self, x):
            x = self.features(x).flatten(1)
            return self.classifier(x)

    model = TinyClassifier()
    x = torch.randn(4, 3, 16, 16)
    y = torch.randint(0, 10, (4,))
    loader = [(x, y)]

    cfg = {
        "backend": "pytorch",
        "model_type": "vgg16",
        "epochs": 0,
        "ft_epochs": 0,
        "ratio": 0.25,
        "method": "l1_norm",
        "scope": "local",
        "input_shape": (3, 16, 16),
        "num_classes": 10,
    }
    orch = Orchestrator(cfg)
    adapter = PyTorchAdapter(cfg)
    pruned_model, masks = orch.run(loader, model=model, adapter=adapter)

    assert pruned_model is not None
    assert isinstance(masks, dict)
    assert len(masks) > 0


def test_orchestrator_can_build_model_from_adapter_only(monkeypatch):
    monkeypatch.setattr("reducnn.engine.orchestrator.plot_training_history", lambda *a, **k: None)
    monkeypatch.setattr("reducnn.engine.orchestrator.plot_layer_sensitivity", lambda *a, **k: None)
    monkeypatch.setattr("reducnn.engine.orchestrator.plot_metrics_comparison", lambda *a, **k: None)

    class TinyAdapter:
        def __init__(self):
            self._model = nn.Sequential(
                nn.Conv2d(3, 4, kernel_size=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(4, 10),
            )

        def get_model(self, model_type):
            return self._model

        def train(self, model, loader, epochs, name, val_loader=None, plot=True):
            return {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        def evaluate(self, model, loader):
            return 0.0

        def get_score_map(self, model, loader, method):
            conv = dict(model.named_modules())["0"]
            w = conv.weight.data.cpu().numpy()
            return {"0": np.mean(np.abs(w), axis=(1, 2, 3))}

        def apply_surgery(self, model, masks):
            return model

        def get_stats(self, model, loader=None):
            return 0.0, float(sum(p.numel() for p in model.parameters()))

        def save_checkpoint(self, model, path):
            return None

        def load_checkpoint(self, model, path):
            return None

        def trace_graph(self, model):
            return {"nodes": {"0": {"type": "conv2d", "inputs": [], "outputs": [], "cluster": None}}, "clusters": {}}

        def classify_architecture(self, model):
            return "sequential"

        def get_multi_metric_scores(self, model, loader, metrics):
            return {}

    x = torch.randn(2, 3, 8, 8)
    y = torch.randint(0, 10, (2,))
    loader = [(x, y)]
    orch = Orchestrator({"epochs": 0, "ft_epochs": 0, "ratio": 0.2, "method": "l1_norm", "scope": "local"})
    model, masks = orch.run(loader, adapter=TinyAdapter())

    assert model is not None
    assert isinstance(masks, dict)


def test_apoz_keeps_active_channels_keras():
    tf = pytest.importorskip("tensorflow")
    from reducnn.backends.keras_backend import KerasAdapter

    inp = tf.keras.Input(shape=(4, 4, 1))
    out = tf.keras.layers.Conv2D(2, 1, use_bias=False, name="conv")(inp)
    model = tf.keras.Model(inp, out)
    weights = np.zeros((1, 1, 1, 2), dtype=np.float32)
    weights[0, 0, 0, 1] = 1.0
    model.get_layer("conv").set_weights([weights])

    x = np.ones((4, 4, 4, 1), dtype=np.float32)
    y = np.zeros((4,), dtype=np.int32)
    loader = [(x, y)]

    adapter = KerasAdapter({"input_shape": (4, 4, 1), "num_classes": 2})
    score = adapter.get_score_map(model, loader, "apoz")["conv"]
    mask = build_pruning_masks({"conv": score}, ratio=0.5, scope="local")["conv"]

    assert bool(mask[1]) is True
    assert bool(mask[0]) is False


def test_hybrid_timing_gate_error_raises_when_ratio_exceeded():
    class DummyAdapter:
        def __init__(self):
            self.config = {
                "hybrid_timing_gate": "error",
                "hybrid_timing_max_ratio": 1.5,
                "hybrid_measure_taylor_baseline": True,
            }

        def trace_graph(self, model):
            return {
                "nodes": {"conv": {"type": "conv2d", "inputs": [], "outputs": [], "cluster": None}},
                "clusters": {},
            }

        def get_score_map(self, model, loader, method):
            time.sleep(0.002)
            return {"conv": np.array([0.2, 0.8], dtype=np.float64)}

        def get_multi_metric_scores(self, model, loader, metrics):
            time.sleep(0.01)
            return {
                "l1_norm": {"conv": np.array([0.2, 0.8], dtype=np.float64)},
                "mean_abs_act": {"conv": np.array([0.4, 0.6], dtype=np.float64)},
                "taylor": {"conv": np.array([0.1, 0.9], dtype=np.float64)},
            }

    meta = HybridMetaPruner(DummyAdapter(), mode="smooth")
    with pytest.raises(RuntimeError):
        _ = meta.calculate_hybrid_scores(model=object(), loader=None)


def test_hybrid_timing_report_and_contributions_recorded():
    class DummyAdapter:
        def __init__(self):
            self.config = {
                "hybrid_timing_gate": "warn",
                "hybrid_timing_max_ratio": 10.0,
                "hybrid_measure_taylor_baseline": True,
            }

        def trace_graph(self, model):
            return {
                "nodes": {
                    "conv_a": {"type": "conv2d", "inputs": [], "outputs": [], "cluster": None},
                    "conv_b": {"type": "conv2d", "inputs": [], "outputs": [], "cluster": None},
                },
                "clusters": {},
            }

        def get_score_map(self, model, loader, method):
            return {
                "conv_a": np.array([0.2, 0.8], dtype=np.float64),
                "conv_b": np.array([0.1, 0.9], dtype=np.float64),
            }

        def get_multi_metric_scores(self, model, loader, metrics):
            return {
                "l1_norm": {
                    "conv_a": np.array([0.2, 0.8], dtype=np.float64),
                    "conv_b": np.array([0.1, 0.9], dtype=np.float64),
                },
                "mean_abs_act": {
                    "conv_a": np.array([0.4, 0.6], dtype=np.float64),
                    "conv_b": np.array([0.7, 0.3], dtype=np.float64),
                },
                "taylor": {
                    "conv_a": np.array([0.1, 0.9], dtype=np.float64),
                    "conv_b": np.array([0.3, 0.7], dtype=np.float64),
                },
            }

    meta = HybridMetaPruner(DummyAdapter(), mode="smooth")
    out = meta.calculate_hybrid_scores(model=object(), loader=None)

    assert "conv_a" in out and "conv_b" in out
    assert "hybrid_time_s" in meta.timing_report
    assert "hybrid_to_taylor_ratio" in meta.timing_report
    assert "conv_a" in meta.last_metric_contributions
    assert "l1_norm" in meta.last_metric_contributions["conv_a"]
