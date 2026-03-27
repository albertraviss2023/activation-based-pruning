import os
from pathlib import Path

import torch

from reducnn.backends.torch_backend import PyTorchAdapter
from reducnn.pruner.mask_builder import build_pruning_masks
from reducnn.visualization.animator import PruningAnimator


def main():
    out_dir = Path("outputs/overhaul_demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = {"input_shape": (3, 32, 32), "num_classes": 10}
    adapter = PyTorchAdapter(cfg)
    model = adapter.get_model("resnet18", pretrained=False)

    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    loader = [(x, y)]

    score_map = adapter.get_score_map(model, loader, "apoz")
    graph = adapter.trace_graph(model)
    masks = build_pruning_masks(score_map, ratio=0.25, scope="local", clusters=graph.get("clusters", {}))

    animator = PruningAnimator(adapter)

    trace = animator.build_pruning_trace(
        model=model,
        score_map=score_map,
        masks=masks,
        method_name="apoz",
        candidate_ratio=0.25,
    )
    trace_path = animator.export_pruning_trace(trace, str(out_dir / "pruning_trace.json"))

    fig_candidates = animator.generate_candidate_discovery_graph(
        model=model,
        score_map=score_map,
        masks=masks,
        method_name="apoz",
        candidate_ratio=0.25,
    )
    candidates_html = animator.export_html(fig_candidates, str(out_dir / "candidate_discovery.html"))

    fig_process = animator.generate_pruning_process_animation(
        model=model,
        score_map=score_map,
        masks=masks,
        method_name="apoz",
        candidate_ratio=0.25,
    )
    process_html = animator.export_html(fig_process, str(out_dir / "pruning_process.html"))

    fig_arch = animator.generate_architecture_comparison(model=model, masks=masks, method_name="apoz")
    arch_html = animator.export_html(fig_arch, str(out_dir / "architecture_comparison.html"))

    print("Overhaul demo exports:")
    print(f"- Trace JSON: {trace_path}")
    print(f"- Candidate discovery: {candidates_html}")
    print(f"- Pruning process animation: {process_html}")
    print(f"- Architecture comparison: {arch_html}")


if __name__ == "__main__":
    main()
