import numpy as np
from typing import Any, Optional
from .registry import register_method

# NOTE:
# The package-level bundled criteria are intentionally minimal:
#   - l1_norm
# Activation-based bundled methods (apoz, mean_abs_act) are implemented in backend adapters.
# Additional methods should be added via custom registration in experiment notebooks.

@register_method("l1_norm")
@register_method("l1")
def l1_norm_score(layer: Any, **kwargs) -> Optional[np.ndarray]:
    """Calculates the L1 norm of filters/channels.

    This scoring method measures filter importance based on the magnitude 
    of its weights. It handles both PyTorch and Keras layer types.

    Args:
        layer (Any): The convolutional layer to score.
        **kwargs: Additional parameters passed from the registry (e.g., 'model').

    Returns:
        np.ndarray: A 1D array of scores, one for each output channel/filter.
    """
    # Heuristic extraction of weights into a NumPy array
    w = _extract_weights(layer)
    if w is None: 
        return None
        
    # Heuristic for determining the framework and appropriate axes for reduction
    # PyTorch Weight Shape: (out_channels, in_channels, height, width) OR (out_f, in_f)
    # Keras Weight Shape: (height, width, in_channels, out_channels) OR (in_f, out_f)
    if "torch" in str(type(layer)).lower():
        # PyTorch layout: (out_channels, in_channels, ...)
        axes = tuple(range(1, w.ndim))
        return np.sum(np.abs(w), axis=axes) if axes else np.abs(w)

    # Keras layout: (..., out_channels)
    axes = tuple(range(w.ndim - 1))
    return np.sum(np.abs(w), axis=axes) if axes else np.abs(w)

def _extract_weights(layer: Any) -> Optional[np.ndarray]:
    """Helper to extract NumPy weight arrays from supported framework layers.

    Args:
        layer (Any): A PyTorch or Keras convolutional layer.

    Returns:
        Optional[np.ndarray]: The weight tensor as a NumPy array, or None if unknown.
    """
    layer_type = str(type(layer)).lower()
    
    if "torch" in layer_type:
        # For PyTorch: layer.weight is a Parameter, .data.cpu() converts to tensor, 
        # then to NumPy.
        return layer.weight.data.cpu().numpy()
        
    elif "keras" in layer_type or "tensorflow" in layer_type:
        # For Keras: layer.get_weights() returns a list [weights, bias]
        weights_list = layer.get_weights()
        return weights_list[0] if weights_list else None
        
    return None
