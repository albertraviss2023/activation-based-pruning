import json
from pathlib import Path

import torch

from reducnn.backends.torch_backend import PyTorchAdapter
from reducnn.pruner import ReduCNNPruner
from reducnn.pruner.mask_builder import build_pruning_masks
from reducnn.visualization.animator import PruningAnimator
from reducnn.visualization.flow_animator import GlobalFlowVisualizer, GlobalMethodComparator


def _mock_loader(batch_size: int = 2, img_size: int = 32, num_classes: int = 10):
    x = torch.randn(batch_size, 3, img_size, img_size)
    y = torch.randint(0, num_classes, (batch_size,))
    return [(x, y)]


def test_overhaul_visual_story_apis(tmp_path):
    cfg = {"input_shape": (3, 32, 32), "num_classes": 10}
    adapter = PyTorchAdapter(cfg)
    model = adapter.get_model("resnet18", pretrained=False)
    loader = _mock_loader()

    pruner = ReduCNNPruner(method="apoz", scope="local")
    _, masks, _ = pruner.prune(model, loader, ratio=0.2, adapter=adapter)
    score_map = adapter.get_score_map(model, loader, "apoz")

    animator = PruningAnimator(adapter)

    trace = animator.build_pruning_trace(
        model=model,
        score_map=score_map,
        masks=masks,
        method_name="apoz",
        candidate_ratio=0.2,
    )
    assert trace["meta"]["node_count"] > 0
    assert "layers" in trace and len(trace["layers"]) > 0
    # Candidate interpretation should respect cluster-harmonized mask decisions.
    for _, members in trace["clusters"].items():
        if len(members) < 2:
            continue
        cand_sets = [set(trace["layers"][m]["candidate_indices"]) for m in members if m in trace["layers"]]
        if len(cand_sets) > 1:
            assert all(s == cand_sets[0] for s in cand_sets[1:])

    out_json = Path(tmp_path) / "trace.json"
    exported = animator.export_pruning_trace(trace, str(out_json))
    assert Path(exported).exists()
    with open(exported, "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert "meta" in payload and "layers" in payload

    fig_candidates = animator.generate_candidate_discovery_graph(
        model=model,
        score_map=score_map,
        masks=masks,
        method_name="apoz",
        candidate_ratio=0.2,
    )
    assert fig_candidates is not None
    assert len(fig_candidates.data) >= 2

    fig_process = animator.generate_pruning_process_animation(
        model=model,
        score_map=score_map,
        masks=masks,
        method_name="apoz",
        candidate_ratio=0.2,
    )
    assert fig_process is not None
    assert len(fig_process.frames) >= 5

    fig_arch = animator.generate_architecture_comparison(model=model, masks=masks, method_name="apoz")
    assert fig_arch is not None
    assert len(fig_arch.data) >= 2
    bullets = animator.summarize_trace_insights(trace, max_lines=5)
    assert isinstance(bullets, list)
    assert len(bullets) >= 3


def test_flow_visualizer_delta_modes_smoke():
    cfg = {"input_shape": (3, 32, 32), "num_classes": 10}
    adapter = PyTorchAdapter(cfg)
    model = adapter.get_model("resnet18", pretrained=False)
    loader = _mock_loader()
    graph = adapter.trace_graph(model)
    acts = adapter.get_global_activations(model, loader)

    sm_l1 = adapter.get_score_map(model, loader, "l1_norm")
    sm_act = adapter.get_score_map(model, loader, "mean_abs_act")
    m1 = build_pruning_masks(sm_l1, ratio=0.3, scope="local", clusters=graph.get("clusters", {}))
    m2 = build_pruning_masks(sm_act, ratio=0.3, scope="local", clusters=graph.get("clusters", {}))

    vis = GlobalFlowVisualizer(
        model_name="resnet18",
        graph=graph,
        activations=acts,
        scores=sm_l1,
        masks=m1,
        delta_ref_masks=m2,
        final_hold_frames=5,
        total_frames=40,
    )
    a0 = vis.update(0)
    a1 = vis.update(45)
    assert a0 is not None and a1 is not None

    comp = GlobalMethodComparator(
        model_name="resnet18",
        graph=graph,
        activations=acts,
        method_a_data={"name": "l1", "scores": sm_l1, "masks": m1},
        method_b_data={"name": "mean", "scores": sm_act, "masks": m2},
        delta_mode=True,
        total_frames=40,
        final_hold_frames=5,
    )
    c0 = comp.update(0)
    c1 = comp.update(45)
    assert c0 is not None and c1 is not None
