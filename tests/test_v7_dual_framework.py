from __future__ import annotations

import os

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from torch.utils.data import DataLoader, TensorDataset

from tests.notebook_utils import load_notebook_namespace


def _load_v7():
    return load_notebook_namespace(
        "generalized_pruning_v7_dual_framework.ipynb",
        "# =========================================\n# 7. MAIN EXECUTION",
    )


def test_v7_pytorch_adapter_core_methods(tmp_path):
    v7 = _load_v7()
    torch = pytest.importorskip("torch")

    cfg = {
        "backend": "pytorch",
        "model_type": "vgg16",
        "lr": 1e-3,
        "epochs": 0,
        "ft_epochs": 0,
        "ratio": 0.4,
        "name": str(tmp_path / "v7_pt_unit"),
        "force_retrain": True,
        "method": "taylor",
    }
    adapter = v7.PyTorchAdapter(cfg)

    model = adapter.get_model("vgg16")
    # Verify BatchNorm parity
    bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert len(bn_layers) == 13, f"PyTorch VGG16 should have 13 BN layers, found {len(bn_layers)}"
    
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    loader = DataLoader(TensorDataset(x, y), batch_size=2)

    adapter.train(model, loader, epochs=0, name="Baseline")
    assert os.path.exists(f"{cfg['name']}_Baseline.pth")

    acc = adapter.evaluate(model, loader)
    assert 0.0 <= acc <= 100.0

    viz = adapter.get_viz_data(model, loader, num_layers=2)
    assert "activations" in viz and "filters" in viz
    assert len(viz["activations"]) == 2
    assert isinstance(viz["filters"], np.ndarray)

    pruned_model, masks = adapter.apply_pruning(model, loader, ratio=0.25, method="taylor")
    assert pruned_model is not None
    assert masks

    v7.plot_viz_suite(viz, "PyTorch Unit")
    v7.plot_comparison((1e6, 2e6), (0.8e6, 1.6e6))


def test_v7_keras_adapter_core_methods_and_viz():
    v7 = _load_v7()
    tf = pytest.importorskip("tensorflow")

    cfg = {
        "backend": "keras",
        "model_type": "vgg16",
        "lr": 1e-3,
        "epochs": 1,
        "ft_epochs": 1,
        "ratio": 0.3,
        "name": "v7_keras_unit",
        "force_retrain": True,
        "method": "apoz",
    }
    adapter = v7.KerasAdapter(cfg)

    # Ensure notebook's Keras model builder still works.
    built_model = adapter.get_model("vgg16")
    assert built_model.output_shape[-1] == 10
    # Verify BatchNorm parity
    bn_layers = [l for l in built_model.layers if "batch_normalization" in l.name.lower() or "bn" in l.name.lower()]
    assert len(bn_layers) >= 13, f"Keras VGG16 should have at least 13 BN layers, found {len(bn_layers)}"

    # Use a tiny model for fast unit tests of train/evaluate/viz.
    inp = tf.keras.Input(shape=(32, 32, 3))
    z = tf.keras.layers.Conv2D(8, 3, activation="relu")(inp)
    z = tf.keras.layers.Conv2D(8, 3, activation="relu")(z)
    z = tf.keras.layers.GlobalAveragePooling2D()(z)
    out = tf.keras.layers.Dense(10, activation="softmax")(z)
    small = tf.keras.Model(inputs=inp, outputs=out)
    small.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    x = np.random.rand(8, 32, 32, 3).astype("float32")
    y = np.random.randint(0, 10, size=(8,)).astype("int32")
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(4)

    adapter.train(small, ds, epochs=1, name="SmallKeras")
    acc = adapter.evaluate(small, ds)
    assert 0.0 <= acc <= 100.0

    viz = adapter.get_viz_data(small, ds, num_layers=2)
    assert "activations" in viz and "filters" in viz
    assert len(viz["activations"]) == 2
    assert viz["filters"].shape[0] == 8

    pruned_model, masks = adapter.apply_pruning(small, ds, ratio=0.25, method="apoz")
    assert pruned_model is not None
    assert masks

    v7.plot_viz_suite(viz, "Keras Unit")
    v7.plot_comparison((1e6, 2e6), (0.7e6, 1.4e6))


def test_researcher_custom_non_notebook_pruning_method():
    v7 = _load_v7()
    torch = pytest.importorskip("torch")

    class ResearchPyTorchAdapter(v7.PyTorchAdapter):
        """Simulate a researcher adding an unlisted pruning method (L2 norm)."""

        def apply_pruning(self, model, loader, ratio, method):
            if method != "l2_norm":
                return super().apply_pruning(model, loader, ratio, method)

            masks = {}
            for name, layer in model.named_modules():
                if isinstance(layer, torch.nn.Conv2d):
                    w = layer.weight.data
                    scores = (w**2).mean(dim=(1, 2, 3))  # L2-energy proxy per filter
                    keep = max(1, int(scores.numel() * (1 - ratio)))
                    idx = torch.topk(scores, keep, largest=True).indices
                    mask = torch.zeros(scores.numel(), dtype=torch.bool)
                    mask[idx] = True
                    masks[name] = mask
            return model, masks

    cfg = {
        "backend": "pytorch",
        "model_type": "vgg16",
        "lr": 1e-3,
        "epochs": 0,
        "ft_epochs": 0,
        "ratio": 0.25,
        "name": "v7_research_unit",
        "force_retrain": True,
        "method": "l2_norm",
    }
    adapter = ResearchPyTorchAdapter(cfg)
    model = adapter.get_model("vgg16")

    x = torch.randn(2, 3, 32, 32)
    y = torch.randint(0, 10, (2,))
    loader = DataLoader(TensorDataset(x, y), batch_size=1)

    _, masks = adapter.apply_pruning(model, loader, ratio=0.25, method="l2_norm")
    assert masks, "Custom research method should produce at least one mask."

    first_mask = next(iter(masks.values()))
    keep_ratio = first_mask.float().mean().item()
    assert 0.6 <= keep_ratio <= 0.9
