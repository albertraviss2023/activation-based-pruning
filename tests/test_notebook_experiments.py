import os
import sys
import torch
import numpy as np
import pytest
from unittest.mock import patch

# Add src to path
sys.path.insert(0, os.path.abspath('src'))

from reducnn.backends.torch_backend import PyTorchAdapter
from reducnn.backends.keras_backend import KerasAdapter
from reducnn.pruner import ReduCNNPruner, register_method
from reducnn.visualization.animator import PruningAnimator


@register_method("chip", framework="torch")
def _chip_torch_test(layer, **kwargs):
    tools = kwargs.get("tools")
    if tools is None:
        return None
    act, _ = tools.collect_layer_outputs(layer, include_labels=False)
    if act is None:
        return None
    return tools.chip_scores(act)


@register_method("chip", framework="keras")
def _chip_keras_test(layer, **kwargs):
    tools = kwargs.get("tools")
    if tools is None:
        return None
    act, _ = tools.collect_layer_outputs(layer, include_labels=False)
    if act is None:
        return None
    return tools.chip_scores(act)

def mock_loader(batch_size, img_size, num_classes, framework='torch'):
    """Creates a mock loader for testing without downloading datasets."""
    if framework == 'torch':
        x = torch.randn(batch_size, 3, img_size, img_size)
        y = torch.randint(0, num_classes, (batch_size,))
        return [(x, y)]
    else:
        # Keras/TF
        x = np.random.randn(batch_size, img_size, img_size, 3).astype(np.float32)
        y = np.random.randint(0, num_classes, (batch_size,))
        return (x, y)

def test_resnet_apoz_experiment_logic():
    """Logic from experiments_cifar10.ipynb - ResNet + APoZ."""
    img_size = 32
    num_classes = 10
    
    adapter = PyTorchAdapter(config={'input_shape': (3, img_size, img_size), 'num_classes': num_classes})
    loader = mock_loader(2, img_size, num_classes)
    
    model = adapter.get_model("resnet18")
    surgeon = ReduCNNPruner(method='apoz', scope='local')
    pruned, masks, dur = surgeon.prune(model, loader, ratio=0.2)
    
    # 1. Verify structural integrity
    x, _ = loader[0]
    out = pruned(x)
    assert out.shape == (2, num_classes)
    
    # 2. Verify parameter reduction
    f1, p1 = adapter.get_stats(model, loader)
    f2, p2 = adapter.get_stats(pruned, loader)
    assert p2 < p1
    
    # 3. Verify visualization
    animator = PruningAnimator(adapter)
    score_map = adapter.get_score_map(model, loader, 'apoz')
    fig = animator.generate_xray_animation(model, score_map, masks)
    assert fig is not None

def test_densenet_chip_experiment_logic():
    """Logic from experiments_cifar100.ipynb - DenseNet + CHIP."""
    img_size = 32
    num_classes = 100
    
    adapter = PyTorchAdapter(config={'input_shape': (3, img_size, img_size), 'num_classes': num_classes})
    loader = mock_loader(2, img_size, num_classes)
    
    model = adapter.get_model("densenet121")
    surgeon = ReduCNNPruner(method='chip', scope='local')
    pruned, masks, dur = surgeon.prune(model, loader, ratio=0.1)
    
    # Verify forward pass
    x, _ = loader[0]
    out = pruned(x)
    assert out.shape == (2, num_classes)
    
    # Verify parameter reduction
    _, p1 = adapter.get_stats(model, loader)
    _, p2 = adapter.get_stats(pruned, loader)
    assert p2 < p1

def test_hybrid_contribution_graph_logic():
    """Validates graph-level hybrid contribution visualization output."""
    img_size = 32
    num_classes = 10

    adapter = PyTorchAdapter(config={'input_shape': (3, img_size, img_size), 'num_classes': num_classes})
    loader = mock_loader(2, img_size, num_classes)
    model = adapter.get_model("resnet18")

    animator = PruningAnimator(adapter)
    fig = animator.generate_hybrid_contribution_graph(model, loader, mode="smooth")

    assert fig is not None
    assert len(fig.data) >= 2

def test_keras_vgg_experiment_logic():
    """Logic from Keras sections of notebooks."""
    img_size = 32
    num_classes = 10
    
    adapter = KerasAdapter(config={'input_shape': (img_size, img_size, 3), 'num_classes': num_classes})
    x, y = mock_loader(2, img_size, num_classes, framework='keras')
    
    model = adapter.get_model("vgg16")
    surgeon = ReduCNNPruner(method='l1_norm', scope='local')
    # Use small ratio for Keras test speed
    pruned, masks, dur = surgeon.prune(model, [(x, y)], ratio=0.1)
    
    # Verify surgery
    out = pruned.predict(x, verbose=0)
    assert out.shape == (2, num_classes)
    
    # Verify parameter reduction
    _, p1 = adapter.get_stats(model)
    _, p2 = adapter.get_stats(pruned)
    assert p2 < p1

def test_cifar100_notebook_full_sequence():
    """Simulates the full research sequence in experiments_cifar100.ipynb."""
    img_size = 32
    num_classes = 100
    adapter = PyTorchAdapter(config={'input_shape': (3, img_size, img_size), 'num_classes': num_classes})
    loader = mock_loader(2, img_size, num_classes)
    
    # 1. VGG16 + L1 Norm
    vgg_model = adapter.get_model("vgg16")
    vgg_pruner = ReduCNNPruner(method='l1_norm', scope='local')
    vgg_pruned, _, _ = vgg_pruner.prune(vgg_model, loader, ratio=0.3)
    assert adapter.get_stats(vgg_pruned)[1] < adapter.get_stats(vgg_model)[1]
    
    # 2. ResNet + APoZ (Global)
    res_model = adapter.get_model("resnet18", pretrained=True)
    res_pruner = ReduCNNPruner(method='apoz', scope='global')
    res_pruned, _, _ = res_pruner.prune(res_model, loader, ratio=0.2)
    assert adapter.classify_architecture(res_model) == 'residual'
    
    # 3. DenseNet + CHIP (Advanced Concept)
    dn_model = adapter.get_model("densenet121", pretrained=True)
    dn_pruner = ReduCNNPruner(method='chip', scope='global')
    dn_pruned, dn_masks, _ = dn_pruner.prune(dn_model, loader, ratio=0.1)
    
    # Verify the specific 0.6.4 metrics and objects
    b_stats = adapter.get_stats(dn_model, loader)
    p_stats = adapter.get_stats(dn_pruned, loader)
    roi = b_stats[1] / p_stats[1]
    assert roi > 1.0
    
    # Verify Visualizer Integration
    animator = PruningAnimator(adapter)
    fig = animator.generate_xray_animation(dn_model, adapter.get_score_map(dn_model, loader, 'chip'), dn_masks)
    assert len(fig.frames) >= 3

def test_cat_dog_custom_model_workflow():
    """Logic from experiments_cat_dog.ipynb - Custom Model Importing."""
    img_size = 128
    num_classes = 2
    
    adapter = PyTorchAdapter(config={'input_shape': (3, img_size, img_size), 'num_classes': num_classes})
    loader = mock_loader(2, img_size, num_classes)
    
    # Simulate a "pretrained" model the user might have
    import torchvision.models as models
    model = models.resnet18(num_classes=num_classes)
    
    surgeon = ReduCNNPruner(method='l1_norm', scope='global')
    # Use the new 0.6.4 prune_custom_model API
    pruned, masks, dur = surgeon.prune_custom_model(model, loader, ratio=0.3)
    
    assert adapter.get_stats(pruned)[1] < adapter.get_stats(model)[1]
    
    # Verify forward pass
    x, _ = loader[0]
    out = pruned(x)
    assert out.shape == (2, num_classes)

def test_keras_densenet_chip_logic():
    """Simulates Keras DenseNet-121 + CHIP workflow."""
    img_size = 32
    num_classes = 10
    adapter = KerasAdapter(config={'input_shape': (img_size, img_size, 3), 'num_classes': num_classes})
    x, y = mock_loader(2, img_size, num_classes, framework='keras')
    
    model = adapter.get_model("densenet121")
    surgeon = ReduCNNPruner(method='chip', scope='global')
    pruned, masks, dur = surgeon.prune(model, [(x, y)], ratio=0.1)
    
    # Verify forward pass
    out = pruned.predict(x, verbose=0)
    assert out.shape == (2, num_classes)
    
    # Verify parameter reduction
    _, p1 = adapter.get_stats(model)
    _, p2 = adapter.get_stats(pruned)
    assert p2 < p1

if __name__ == "__main__":
    pytest.main([__file__, "-s"])
