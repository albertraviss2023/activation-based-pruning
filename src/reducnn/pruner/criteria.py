import numpy as np
from typing import Dict, Any, Optional
from .registry import register_method

# Pruning criteria are implemented using framework-agnostic logic where possible.
# We utilize the 'register_method' decorator to populate the global registry.

@register_method("l1_norm")
@register_method("l1")
def l1_norm_score(layer: Any, **kwargs) -> Optional[np.ndarray]:
    """Calculates the average L1 norm (Mean Absolute Value) of filters.

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
    # PyTorch Weight Shape: (out_channels, in_channels, height, width)
    # Keras Weight Shape: (height, width, in_channels, out_channels)
    if "torch" in str(type(layer)).lower():
        # Reduce over (in_channels, h, w) - dims (1, 2, 3)
        return np.mean(np.abs(w), axis=(1, 2, 3))
    else:
        # Reduce over (h, w, in_channels) - dims (0, 1, 2)
        return np.mean(np.abs(w), axis=(0, 1, 2))

@register_method("l2_norm")
@register_method("l2")
def l2_norm_score(layer: Any, **kwargs) -> Optional[np.ndarray]:
    """Calculates the L2 norm (Root Mean Square) of filters.

    Args:
        layer (Any): The convolutional layer to score.
        **kwargs: Additional parameters.

    Returns:
        np.ndarray: A 1D array of importance scores.
    """
    w = _extract_weights(layer)
    if w is None: 
        return None
        
    if "torch" in str(type(layer)).lower():
        # RMS over dims (1, 2, 3)
        return np.sqrt(np.mean(np.square(w), axis=(1, 2, 3)) + 1e-12)
    else:
        # RMS over dims (0, 1, 2)
        return np.sqrt(np.mean(np.square(w), axis=(0, 1, 2)) + 1e-12)

@register_method("random")
@register_method("rand")
def random_score(layer: Any, **kwargs) -> Optional[np.ndarray]:
    """Assigns random scores to filters for baseline testing.

    Args:
        layer (Any): The layer.
        **kwargs: Additional parameters.

    Returns:
        np.ndarray: A 1D array of random scores.
    """
    w = _extract_weights(layer)
    if w is None: 
        return None
        
    if "torch" in str(type(layer)).lower():
        return np.random.rand(w.shape[0])
    else:
        return np.random.rand(w.shape[-1])

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

@register_method("taylor")
def taylor_score(layer: Any, model: Any, loader: Any, device: Any = None, **kwargs) -> np.ndarray:
    """Calculates the Taylor-1 importance score (Abs(Activation * Gradient)).

    This method requires forward and backward passes. Because efficient 
    implementation involves complex hook management, this function 
    is a placeholder that directs users to framework-native implementations.

    Args:
        layer (Any): Target layer.
        model (Any): Full model.
        loader (Any): Calibration data.
        device (Any, optional): Device context. Defaults to None.
        **kwargs: Additional parameters.

    Raises:
        NotImplementedError: As this method is handled natively by Backend Adapters.
    """
    raise NotImplementedError(
        "Taylor pruning is handled natively by the Backend Adapters "
        "for efficiency due to deep hook/gradient dependencies."
    )
