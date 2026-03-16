import os
import sys
import numpy as np
from typing import Dict, Any

# Add src to path for direct package testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from reducnn.pruner.registry import register_method
from reducnn.pruner import ReduCNNPruner

# Register framework-specific test methods
@register_method("custom_l2_test", framework="torch")
def custom_l2_score_torch(layer, **kwargs):
    w = layer.weight.data.cpu().numpy()
    return np.sqrt(np.mean(np.square(w), axis=(1, 2, 3)) + 1e-12)

@register_method("custom_l2_test", framework="keras")
def custom_l2_score_keras(layer, **kwargs):
    w = layer.get_weights()[0]
    return np.sqrt(np.mean(np.square(w), axis=(0, 1, 2)) + 1e-12)

@register_method("chip_test", framework="global")
def chip_math_global(layer, **kwargs):
    """Fallback/Global implementation for test."""
    model = kwargs.get('model')
    loader = kwargs.get('loader')
    if model is None or loader is None: return None
    
    l_type = str(type(layer)).lower()
    if "torch" in l_type:
        out_channels = layer.out_channels
    else:
        out_channels = layer.filters
    return np.random.rand(out_channels)

def test_custom_methods_keras():
    try:
        import tensorflow as tf
    except ImportError:
        print("Skipping Keras tests: TensorFlow not installed.")
        return

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(8, 3, padding='same', input_shape=(32, 32, 3)),
        tf.keras.layers.ReLU()
    ])
    # Mock loader
    x = np.random.randn(2, 32, 32, 3).astype(np.float32)
    y = np.random.randint(0, 10, (2,)).astype(np.int32)
    loader = [(x, y)]
    
    # Test Custom L2 (Keras specific registration)
    print("Testing Keras Custom L2 (Framework-Specific Registration)...")
    surgeon = ReduCNNPruner(method='custom_l2_test')
    pruned, masks, duration = surgeon.prune(model, loader, ratio=0.5)
    assert len(masks) > 0
    print("Keras Custom L2 test passed!")

    # Test CHIP (Global registration)
    print("Testing Keras Custom CHIP (Global Fallback)...")
    surgeon = ReduCNNPruner(method='chip_test')
    pruned, masks, duration = surgeon.prune(model, loader, ratio=0.5)
    assert len(masks) > 0
    print("Keras Custom CHIP test passed!")

def test_custom_methods_pytorch():
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("Skipping PyTorch tests: Torch not installed.")
        return

    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU()
    )
    # Mock loader
    x = torch.randn(2, 3, 32, 32)
    y = torch.randint(0, 10, (2,))
    loader = [(x, y)]
    
    # Test Custom L2 (Torch specific registration)
    print("Testing PyTorch Custom L2 (Framework-Specific Registration)...")
    surgeon = ReduCNNPruner(method='custom_l2_test')
    pruned, masks, duration = surgeon.prune(model, loader, ratio=0.5)
    assert len(masks) > 0
    assert "0" in masks
    print("PyTorch Custom L2 test passed!")

    # Test CHIP (Global registration)
    print("Testing PyTorch Custom CHIP (Global Fallback)...")
    surgeon = ReduCNNPruner(method='chip_test')
    pruned, masks, duration = surgeon.prune(model, loader, ratio=0.5)
    assert len(masks) > 0
    assert "0" in masks
    print("PyTorch Custom CHIP test passed!")

if __name__ == "__main__":
    try:
        test_custom_methods_keras()
        test_custom_methods_pytorch()
        print("\n✅ All unit tests on custom methods passed!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ Tests failed: {e}")
        sys.exit(1)
