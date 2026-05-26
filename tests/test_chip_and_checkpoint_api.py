from __future__ import annotations

import os
import sys

import numpy as np

# Add src to path for direct package testing
sys.path.insert(0, os.path.abspath("src"))

from reducnn.pruner.chip import chip_channel_independence_scores
from reducnn.pruner.surgeon import ReduCNNPruner


def test_chip_scores_are_layout_invariant_between_nchw_and_nhwc():
    rng = np.random.default_rng(42)
    nchw = rng.normal(size=(3, 8, 6, 6)).astype(np.float32)
    nhwc = np.moveaxis(nchw, 1, -1)

    s_nchw = chip_channel_independence_scores(nchw, channel_axis=1, max_spatial=32)
    s_nhwc = chip_channel_independence_scores(nhwc, channel_axis=-1, max_spatial=32)

    assert s_nchw.shape == (8,)
    assert np.allclose(s_nchw, s_nhwc, atol=1e-9)


def test_prune_custom_model_supports_checkpoint_load_and_pruned_save(tmp_path):
    class DummyAdapter:
        def __init__(self):
            self.loaded = []
            self.saved = []

        def trace_graph(self, _model):
            return {
                "nodes": {"conv": {"type": "conv2d", "inputs": [], "outputs": [], "cluster": None}},
                "clusters": {},
            }

        def classify_architecture(self, _model):
            return "sequential"

        def get_score_map(self, _model, _loader, _method):
            return {"conv": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)}

        def apply_surgery(self, _model, _masks):
            return {"pruned": True}

        def load_checkpoint(self, _model, path: str):
            self.loaded.append(path)

        def save_checkpoint(self, _model, path: str):
            self.saved.append(path)

    adapter = DummyAdapter()
    surgeon = ReduCNNPruner(method="l1_norm", scope="local")

    save_path = tmp_path / "exports" / "dummy_pruned.ckpt"
    pruned, masks, _ = surgeon.prune_custom_model(
        model=object(),
        loader=None,
        ratio=0.5,
        adapter=adapter,
        checkpoint_path="repo/pretrained.ckpt",
        save_pruned_path=str(save_path),
    )

    assert pruned == {"pruned": True}
    assert "conv" in masks
    assert adapter.loaded == ["repo/pretrained.ckpt"]
    assert adapter.saved == [str(save_path)]
    assert save_path.parent.exists()
