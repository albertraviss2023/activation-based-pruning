import os
import torch
import numpy as np
from reducnn.backends.factory import get_adapter
from reducnn.pruner import ReduCNNPruner
from reducnn.visualization import GlobalFlowVisualizer, GlobalMethodComparator

def generate_demo():
    print("🚀 Initializing ReduCNN PhD-Level Demo...")
    config = {'backend': 'pytorch', 'model_type': 'vgg11', 'input_shape': (3, 32, 32), 'num_classes': 10}
    adapter = get_adapter(None, config)
    model = adapter.get_model('vgg11', pretrained=False)
    
    # 1. Create Dummy Data
    loader = [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,)))]
    
    # 2. Extract Logic for Method A (L1-Norm)
    print("🔍 Analyzing Method A (L1-Norm)...")
    surgeon_a = ReduCNNPruner(method='l1_norm', scope='local')
    _, masks_a, _ = surgeon_a.prune(model, loader, ratio=0.4)
    scores_a = adapter.get_score_map(model, loader, 'l1_norm')
    
    # 3. Extract Logic for Method B (Mean-Abs-Act)
    print("🔍 Analyzing Method B (Mean-Abs-Act)...")
    surgeon_b = ReduCNNPruner(method='mean_abs_act', scope='local')
    _, masks_b, _ = surgeon_b.prune(model, loader, ratio=0.4)
    scores_b = adapter.get_score_map(model, loader, 'mean_abs_act')
    
    # 4. Shared Data for Pulse
    graph = adapter.trace_graph(model)
    activations = adapter.get_global_activations(model, loader)
    
    # 5. Generate Flagship Surgery Animation
    print("🎬 Rendering Flagship Network Surgery (Narrative Mode)...")
    flow_vis = GlobalFlowVisualizer(
        model_name="VGG11-Surgery",
        graph=graph,
        activations=activations,
        scores=scores_a,
        masks=masks_a,
        out_path="outputs/demo_network_surgery.gif"
    )
    flow_vis.animate()
    
    # 6. Generate Method Battle Animation
    print("🎬 Rendering Method Battle (L1 vs Mean-Act)...")
    battle = GlobalMethodComparator(
        model_name="VGG11-Battle",
        graph=graph,
        activations=activations,
        method_a_data={'name': 'L1-Norm (Weight)', 'scores': scores_a, 'masks': masks_a},
        method_b_data={'name': 'Mean-Act (Data)', 'scores': scores_b, 'masks': masks_b},
        out_path="outputs/demo_method_battle.gif"
    )
    battle.animate()
    
    print("\n✅ DEMO ASSETS READY:")
    print(f"1. Network Surgery: {os.path.abspath('outputs/demo_network_surgery.gif')}")
    print(f"2. Method Battle:   {os.path.abspath('outputs/demo_method_battle.gif')}")

if __name__ == "__main__":
    generate_demo()
