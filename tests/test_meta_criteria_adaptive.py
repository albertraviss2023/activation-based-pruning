from __future__ import annotations

import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

plotly = types.ModuleType("plotly")
plotly_graph_objects = types.ModuleType("plotly.graph_objects")
plotly_subplots = types.ModuleType("plotly.subplots")
plotly_graph_objects.Figure = object
plotly_graph_objects.__getattr__ = lambda name: object
plotly_subplots.make_subplots = lambda *args, **kwargs: object()
plotly.graph_objects = plotly_graph_objects
plotly.subplots = plotly_subplots
sys.modules.setdefault("plotly", plotly)
sys.modules.setdefault("plotly.graph_objects", plotly_graph_objects)
sys.modules.setdefault("plotly.subplots", plotly_subplots)

from reducnn.pruner.custom_method_tools import CustomMethodTools
from reducnn.pruner.meta_criteria import HybridMetaPruner


def test_tis_threshold_aggregate_defines_eps():
    mat = np.asarray(
        [
            [0.1, 0.9, 0.2],
            [0.8, 0.4, 0.3],
        ],
        dtype=np.float64,
    )

    scores = CustomMethodTools.tis_threshold_aggregate(mat, percentile=50.0)

    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))
    assert np.max(scores) > 0.0


def test_adaptive_hybrid_selects_cheap_correlated_proxy():
    class DummyAdapter:
        config = {
            "hybrid_metric_pool": ["l1_norm", "taylor", "chip"],
            "hybrid_correlation_threshold": 0.95,
            "hybrid_topk_overlap_threshold": 0.80,
            "ratio": 0.50,
        }

        def trace_graph(self, model):
            return {"nodes": {"conv": {"type": "conv2d"}}, "clusters": {}}

        def get_multi_metric_scores(self, model, loader, metrics):
            return {
                "l1_norm": {"conv": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)},
                "taylor": {"conv": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)},
                "chip": {"conv": np.array([4.0, 1.0, 3.0, 2.0], dtype=np.float64)},
            }

    meta = HybridMetaPruner(DummyAdapter(), mode="adaptive")
    out = meta.calculate_hybrid_scores(model=object(), loader=None)

    assert "conv" in out
    assert meta.last_layer_decisions["conv"]["mode"] == "cheap_proxy"
    assert meta.last_layer_decisions["conv"]["selected"] == "l1_norm"
    assert "taylor" in meta.last_layer_decisions["conv"]["covered_complex_methods"]
    assert meta.last_layer_decisions["conv"]["similarity_rule"] == "spearman_rank_correlation_and_prune_set_overlap"


def test_adaptive_hybrid_blends_when_no_proxy_matches():
    class DummyAdapter:
        config = {
            "hybrid_metric_pool": ["l1_norm", "taylor", "chip"],
            "hybrid_correlation_threshold": 0.99,
            "hybrid_topk_overlap_threshold": 0.80,
            "ratio": 0.50,
        }

        def trace_graph(self, model):
            return {"nodes": {"conv": {"type": "conv2d"}}, "clusters": {}}

        def get_multi_metric_scores(self, model, loader, metrics):
            return {
                "l1_norm": {"conv": np.array([0.1, 0.2, 0.9, 0.4], dtype=np.float64)},
                "taylor": {"conv": np.array([0.7, 0.2, 0.3, 0.8], dtype=np.float64)},
                "chip": {"conv": np.array([0.4, 0.9, 0.1, 0.5], dtype=np.float64)},
            }

    meta = HybridMetaPruner(DummyAdapter(), mode="adaptive")
    out = meta.calculate_hybrid_scores(model=object(), loader=None)

    assert "conv" in out
    assert meta.last_layer_decisions["conv"]["mode"] == "cost_aware_blend"
    assert set(meta.last_layer_decisions["conv"]["weights"]) == {"l1_norm", "taylor", "chip"}


def test_adaptive_hybrid_can_use_measured_efficiency_records():
    class DummyAdapter:
        config = {
            "backend": "pytorch",
            "model_type": "toy",
            "dataset": "cifar-10",
            "hybrid_efficiency_records": [
                {"backend": "pytorch", "model": "toy", "dataset": "cifar-10", "method": "l1_norm", "status": "ok", "simplicity_time_sec": 10.0},
                {"backend": "pytorch", "model": "toy", "dataset": "cifar-10", "method": "taylor", "status": "ok", "simplicity_time_sec": 40.0},
            ],
        }

    meta = HybridMetaPruner(DummyAdapter(), mode="adaptive")
    costs = meta._load_measured_method_costs()

    assert costs["l1_norm"] == 1.0
    assert costs["taylor"] == 4.0
    assert meta._method_cost("taylor") == 4.0


def test_adaptive_hybrid_stacks_simple_only_methods_by_default():
    class DummyAdapter:
        config = {
            "hybrid_metric_pool": ["l1_norm", "custom_l2", "apoz"],
            "hybrid_correlation_threshold": 0.55,
            "ratio": 0.50,
        }

        def trace_graph(self, model):
            return {"nodes": {"conv": {"type": "conv2d"}}, "clusters": {}}

        def get_multi_metric_scores(self, model, loader, metrics):
            return {
                "l1_norm": {"conv": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)},
                "custom_l2": {"conv": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)},
                "apoz": {"conv": np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)},
            }

    meta = HybridMetaPruner(DummyAdapter(), mode="adaptive")
    _ = meta.calculate_hybrid_scores(model=object(), loader=None)

    decision = meta.last_layer_decisions["conv"]
    assert decision["mode"] == "simple_only_stack"
    assert decision["selected"] == "blend"
    assert set(decision["stack_methods"]) == {"l1_norm", "custom_l2", "apoz"}
    assert set(decision["weights"]) == {"l1_norm", "custom_l2", "apoz"}
    assert np.isclose(sum(decision["weights"].values()), 1.0)


def test_adaptive_hybrid_can_disable_simple_only_stacks():
    class DummyAdapter:
        config = {
            "hybrid_metric_pool": ["l1_norm", "custom_l2", "apoz"],
            "hybrid_correlation_threshold": 0.55,
            "hybrid_allow_simple_only_stack": False,
            "ratio": 0.50,
        }

        def trace_graph(self, model):
            return {"nodes": {"conv": {"type": "conv2d"}}, "clusters": {}}

        def get_multi_metric_scores(self, model, loader, metrics):
            return {
                "l1_norm": {"conv": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)},
                "custom_l2": {"conv": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)},
                "apoz": {"conv": np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)},
            }

    meta = HybridMetaPruner(DummyAdapter(), mode="adaptive")
    _ = meta.calculate_hybrid_scores(model=object(), loader=None)

    decision = meta.last_layer_decisions["conv"]
    assert decision["mode"] == "simple_only_single"
    assert list(decision["weights"].values()) == [1.0]


def test_adaptive_hybrid_blend_uses_at_most_one_simple_representative():
    class DummyAdapter:
        config = {
            "hybrid_metric_pool": ["l1_norm", "custom_l2", "taylor", "chip"],
            "hybrid_correlation_threshold": 0.99,
            "hybrid_topk_overlap_threshold": 0.99,
            "ratio": 0.50,
        }

        def trace_graph(self, model):
            return {"nodes": {"conv": {"type": "conv2d"}}, "clusters": {}}

        def get_multi_metric_scores(self, model, loader, metrics):
            return {
                "l1_norm": {"conv": np.array([0.1, 0.7, 0.3, 0.4], dtype=np.float64)},
                "custom_l2": {"conv": np.array([0.6, 0.2, 0.8, 0.5], dtype=np.float64)},
                "taylor": {"conv": np.array([0.7, 0.2, 0.3, 0.8], dtype=np.float64)},
                "chip": {"conv": np.array([0.4, 0.9, 0.1, 0.5], dtype=np.float64)},
            }

    meta = HybridMetaPruner(DummyAdapter(), mode="adaptive")
    _ = meta.calculate_hybrid_scores(model=object(), loader=None)

    stack_methods = meta.last_layer_decisions["conv"]["stack_methods"]
    simple_in_stack = [m for m in stack_methods if m in {"l1_norm", "custom_l2", "apoz", "mean_abs_act"}]
    assert len(simple_in_stack) <= 1
    assert {"taylor", "chip"}.issubset(set(stack_methods))
