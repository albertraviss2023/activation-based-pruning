import os
import torch
import numpy as np
import tensorflow as tf
from reducnn.backends.factory import get_adapter
from reducnn.pruner import ReduCNNPruner
from reducnn.visualization import (
    plot_layer_sensitivity, 
    plot_metrics_comparison, 
    plot_training_history,
    plot_rank_correlation,
    plot_score_distributions,
    plot_feature_maps,
    plot_decision_agreement,
    plot_inference_gallery,
    PruningAnimator,
    PruningVisualizer
)

def run_agnostic_viz_test(backend, model_type, dataset_name, input_shape, num_classes):
    print(f"\n--- Testing Agnostic Viz: {backend.upper()} | {model_type} | {dataset_name} ---")
    run_id = f"test_{backend}_{model_type}_{dataset_name.lower()}"
    
    config = {
        'backend': backend,
        'model_type': model_type,
        'input_shape': input_shape,
        'num_classes': num_classes,
        'experiment_id': run_id
    }

    # 1. Load Model
    if backend == 'pytorch':
        from reducnn.backends.torch_backend import PyTorchAdapter
        adapter = PyTorchAdapter(config)
        model = adapter.get_model(model_type, pretrained=False) # False for speed
    else:
        from reducnn.backends.keras_backend import KerasAdapter
        adapter = KerasAdapter(config)
        model = adapter.get_model(model_type)
    
    # 2. Create Dummy Data (Agnostic to real dataset loading for this test)
    batch_size = 4
    if backend == 'pytorch':
        dummy_x = torch.randn(batch_size, *input_shape)
        dummy_y = torch.randint(0, num_classes, (batch_size,))
        loader = [(dummy_x, dummy_y)]
    else:
        # Keras expects (H, W, C) usually, adapter handles conversion if we pass (C, H, W)
        # But for dummy data creation we should be careful. 
        # The adapter.get_model already normalized the model's input_shape.
        k_shape = model.input_shape[1:]
        dummy_x = np.random.randn(batch_size, *k_shape).astype(np.float32)
        dummy_y = np.random.randint(0, num_classes, (batch_size,))
        loader = tf.data.Dataset.from_tensor_slices((dummy_x, dummy_y)).batch(batch_size)

    # 3. Tier 1: Stakeholder
    print("  Tier 1: Stakeholder...")
    history = {'train_loss': [0.1], 'train_acc': [90]}
    # plot_training_history(history, "Test") # Skip plt.show() in automated tests
    
    surgeon = ReduCNNPruner(method='l1_norm', scope='local')
    _, masks, _ = surgeon.prune(model, loader, ratio=0.2)
    # plot_layer_sensitivity(masks, "Test")

    # 4. Tier 2: Research
    print("  Tier 2: Research...")
    methods = ['l1_norm', 'mean_abs_act']
    score_maps = {}
    for m in methods:
        score_maps[m] = adapter.get_score_map(model, loader, m)
    
    # plot_score_distributions(score_maps)
    # plot_decision_agreement(score_maps, ratio=0.2)

    # 5. Tier 3: X-Ray & Animation (Files creation check)
    print("  Tier 3: X-Ray & Animations...")
    pvis = PruningVisualizer(model_type, backend, experiment_id=run_id)
    
    target_layer = list(masks.keys())[0]
    activations = adapter.get_layer_activations(model, loader, target_layer)
    
    layer_vis = {
        "layer_name": target_layer,
        "num_channels": len(masks[target_layer]),
        "importance_scores": score_maps['l1_norm'][target_layer],
        "activation_stats": np.mean(activations, axis=0),
        "pruned_mask": (masks[target_layer] == 0)
    }
    
    # We actually save these to verify pathing and naming logic
    pvis.animate_activation_flow(layer_vis, activations[:2], filename="flow.gif")
    pvis.animate_pruning(layer_vis, filename="prune.gif")

    # 6. Verification
    expected_files = [
        f"outputs/{run_id}/activation_flow/flow.gif",
        f"outputs/{run_id}/pruning/prune.gif"
    ]
    
    for f in expected_files:
        if os.path.exists(f):
            print(f"  ✅ Export verified: {f}")
        else:
            raise FileNotFoundError(f"❌ Export failed: {f}")

if __name__ == "__main__":
    # Test Case 1: PyTorch + ResNet + CIFAR shape
    run_agnostic_viz_test('pytorch', 'resnet18', 'CIFAR-10', (3, 32, 32), 10)
    
    # Test Case 2: Keras + VGG + CIFAR shape
    run_agnostic_viz_test('keras', 'vgg16', 'CIFAR-10', (3, 32, 32), 10)
    
    # Test Case 3: PyTorch + MobileNet + High-res shape (agnostic check)
    run_agnostic_viz_test('pytorch', 'mobilenet_v2', 'Custom-Data', (3, 224, 224), 100)

    print("\n🚀 ALL AGNOSTIC VISUALIZATION TESTS PASSED!")
