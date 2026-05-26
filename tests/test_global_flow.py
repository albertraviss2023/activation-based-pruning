import os
import torch
from reducnn.backends.factory import get_adapter
from reducnn.pruner import ReduCNNPruner
from reducnn.visualization import GlobalFlowVisualizer

def run_test():
    print("🚀 Testing GlobalFlowVisualizer...")
    
    config = {
        'backend': 'pytorch',
        'model_type': 'vgg11',
        'input_shape': (3, 32, 32),
        'num_classes': 10
    }

    from reducnn.backends.torch_backend import PyTorchAdapter
    adapter = PyTorchAdapter(config)
    model = adapter.get_model('vgg11', pretrained=False)
    
    batch_size = 2
    dummy_x = torch.randn(batch_size, 3, 32, 32)
    dummy_y = torch.randint(0, 10, (batch_size,))
    loader = [(dummy_x, dummy_y)]

    print("1. Tracing Graph...")
    graph = adapter.trace_graph(model)
    
    print("2. Getting Global Activations...")
    activations = adapter.get_global_activations(model, loader)
    
    print("3. Getting Importance Scores...")
    scores = adapter.get_score_map(model, loader, 'l1_norm')
    
    print("4. Getting Pruning Masks...")
    surgeon = ReduCNNPruner(method='l1_norm', scope='local')
    _, masks, _ = surgeon.prune(model, loader, ratio=0.3)
    
    out_path = "outputs/test_global_flow.gif"
    
    print("5. Generating Animation...")
    visualizer = GlobalFlowVisualizer(
        model_name="VGG-11",
        graph=graph,
        activations=activations,
        scores=scores,
        masks=masks,
        out_path=out_path
    )
    visualizer.animate()
    
    if os.path.exists(out_path):
        print(f"✅ Success: {out_path} created.")
    else:
        print(f"❌ Failed to create {out_path}.")

if __name__ == "__main__":
    run_test()
