"""Example custom methods loaded by ReduCNN Studio."""

from __future__ import annotations

import numpy as np

from reducnn.pruner import register_method


METHOD_METADATA = {
    "custom_activation_variance": {
        "label": "Activation variance",
        "description": "Example custom method that keeps channels with high calibration activation variance.",
    }
}


@register_method("custom_activation_variance", framework="global")
def activation_variance(layer, tools=None, **kwargs):
    """Keep channels whose activations vary most on calibration data."""
    if tools is None:
        return None
    act, _ = tools.collect_layer_outputs(layer, include_labels=False)
    if act is None:
        return tools.weight_l2(layer)
    x = tools.channel_matrix(act)
    return np.var(x, axis=1)
