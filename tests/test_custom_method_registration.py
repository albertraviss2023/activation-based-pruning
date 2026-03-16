import os
import sys
import numpy as np
from typing import Dict, Any

# Add src to path for direct package testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from surgical_pruning.pruner.registry import register_method
from surgical_pruning.pruner import SurgicalPruner

@register_method("custom_l2_test")
def custom_l2_score(layer, **kwargs):
    import numpy as np
    l_type = str(type(layer)).lower()
    if "torch" in l_type:
        w = layer.weight.data.cpu().numpy()
        return np.sqrt(np.mean(np.square(w), axis=(1, 2, 3)) + 1e-12)
    else:
        w = layer.get_weights()[0]
        return np.sqrt(np.mean(np.square(w), axis=(0, 1, 2)) + 1e-12)

@register_method("chip_test")
def chip_math_test(layer, **kwargs):
    model = kwargs.get('model')
    loader = kwargs.get('loader')
    if model is None or loader is None: return None
    
    # Simulate Nuclear Norm for stability
    l_type = str(type(layer)).lower()
    if "torch" in l_type:
        out_channels = layer.out_channels
    else:
        out_channels = layer.filters
    return np.random.rand(out_channels) # Use rand to ensure ranking works

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
    
    # Test Custom L2
    print("Testing Keras Custom L2...")
    surgeon = SurgicalPruner(method='custom_l2_test')
    pruned, masks, duration = surgeon.prune(model, loader, ratio=0.5)
    assert len(masks) > 0
    print("Keras Custom L2 test passed!")

    # Test CHIP (kwargs pass-through)
    print("Testing Keras Custom CHIP...")
    surgeon = SurgicalPruner(method='chip_test')
    pruned, masks, duration = surgeon.prune(model, loader, ratio=0.5)
    assert len(masks) > 0
    print("Keras Custom CHIP test passed!")

if __name__ == "__main__":
    try:
        test_custom_methods_keras()
        print("\n✅ All unit tests on custom methods passed!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ Tests failed: {e}")
        sys.exit(1)
