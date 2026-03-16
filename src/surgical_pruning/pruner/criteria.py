import numpy as np
from typing import Dict
from .registry import register_method

# Ensure we use NumPy operations safely
# We don't want to enforce torch/keras imports here to keep the registry clean.
# Implementations will check types if needed.

@register_method("l1_norm")
@register_method("l1")
def l1_norm_score(layer, **kwargs) -> np.ndarray:
    """Calculates the L1 norm of filters."""
    # Heuristic extraction of weights
    w = _extract_weights(layer)
    if w is None: return None
    # PyTorch: (out, in, h, w). Keras: (h, w, in, out)
    if "torch" in str(type(layer)).lower():
        # returns sum over in, h, w (dims 1, 2, 3)
        return np.mean(np.abs(w), axis=(1, 2, 3))
    else:
        # Keras returns sum over h, w, in (dims 0, 1, 2)
        return np.mean(np.abs(w), axis=(0, 1, 2))

@register_method("l2_norm")
@register_method("l2")
def l2_norm_score(layer, **kwargs) -> np.ndarray:
    """Calculates the L2 norm of filters."""
    w = _extract_weights(layer)
    if w is None: return None
    if "torch" in str(type(layer)).lower():
        return np.sqrt(np.mean(np.square(w), axis=(1, 2, 3)) + 1e-12)
    else:
        return np.sqrt(np.mean(np.square(w), axis=(0, 1, 2)) + 1e-12)

@register_method("random")
@register_method("rand")
def random_score(layer, **kwargs) -> np.ndarray:
    """Assigns random scores for baseline testing."""
    w = _extract_weights(layer)
    if w is None: return None
    if "torch" in str(type(layer)).lower():
        return np.random.rand(w.shape[0])
    else:
        return np.random.rand(w.shape[-1])

def _extract_weights(layer) -> np.ndarray:
    """Helper to pull numpy weights out of either a Torch or Keras layer."""
    layer_type = str(type(layer)).lower()
    if "torch" in layer_type:
        return layer.weight.data.cpu().numpy()
    elif "keras" in layer_type or "tensorflow" in layer_type:
        return layer.get_weights()[0]
    return None

@register_method("taylor")
def taylor_score(layer, model, loader, device=None, **kwargs) -> np.ndarray:
    """
    Calculates the Taylor-1 importance score (Abs(Activation * Gradient)).
    Requires forward and backward passes.
    """
    # This is a proxy implementation. In a real package, the backend adapter 
    # would likely handle the complex hook management for Taylor.
    # Here we assume the adapter passes the scores or we use the logic from V7.
    # To keep this modular, we'll let the user know this is framework-dependent.
    
    # If the user is using the standard adapters, they'll call get_score_map 
    # which has native Taylor support for efficiency.
    raise NotImplementedError("Taylor pruning is handled natively by the Backend Adapters for efficiency.")
