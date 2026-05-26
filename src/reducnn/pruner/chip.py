from __future__ import annotations

from typing import Optional, Union

import numpy as np


def chip_channel_independence_scores(
    activations: Union[np.ndarray, "object"],
    channel_axis: Optional[int] = None,
    max_spatial: Optional[int] = 196,
    eps: float = 1e-8,
) -> np.ndarray:
    """Computes CHIP channel-independence scores from activations.

    The score for each channel is the nuclear-norm change of the layer's
    activation matrix when that channel is removed. Larger change means the
    channel contributes more independent information and should be kept.

    Args:
        activations: Activation tensor/array. Supported:
            - 4D feature maps (NCHW or NHWC)
            - 2D activations (N, C) / (C, N) for non-spatial layers.
        channel_axis: Channel dimension index.
            - For 4D: typically 1 (NCHW) or -1 (NHWC).
            - For 2D: 1 / -1 means channels in last dim (N, C), 0 means (C, N).
            - If None, defaults to 1 for 4D and 1 for 2D.
        max_spatial: Optional cap on spatial samples per image for efficiency (4D only).
        eps: Numerical stability floor.

    Returns:
        np.ndarray: 1D score vector (higher means more independent/important).
    """
    def _nuclear_change(ch_by_feat: np.ndarray) -> np.ndarray:
        x = np.asarray(ch_by_feat, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D channel-feature matrix, got shape={x.shape}.")
        c = int(x.shape[0])
        if c <= 0:
            return np.zeros((0,), dtype=np.float64)
        if c == 1:
            return np.ones((1,), dtype=np.float64)

        # Samples/features by channels, centered per channel.
        m = x.T
        m = m - m.mean(axis=0, keepdims=True)
        base = float(np.sum(np.linalg.svd(m, full_matrices=False, compute_uv=False)))
        scores = np.zeros((c,), dtype=np.float64)
        for i in range(c):
            m_minus = np.delete(m, i, axis=1)
            if m_minus.shape[1] == 0:
                scores[i] = base
            else:
                minus = float(np.sum(np.linalg.svd(m_minus, full_matrices=False, compute_uv=False)))
                scores[i] = max(base - minus, 0.0)
        scale = np.max(scores)
        if scale > eps:
            scores = scores / scale
        return scores.reshape(-1)

    a = np.asarray(activations, dtype=np.float64)
    if a.ndim < 2:
        raise ValueError(f"CHIP expects at least 2D activations, got shape={a.shape}.")

    if channel_axis is None:
        channel_axis = 1
    if channel_axis < 0:
        channel_axis += a.ndim

    if channel_axis < 0 or channel_axis >= a.ndim:
        raise ValueError(f"Invalid channel_axis={channel_axis} for shape={a.shape}.")

    # 2D path (e.g., Linear / Dense activations)
    if a.ndim == 2:
        if channel_axis in (1, -1):
            # (N, C) -> (C, N)
            x = a.T
        elif channel_axis == 0:
            # already (C, N)
            x = a
        else:
            raise ValueError(f"Invalid channel_axis={channel_axis} for 2D shape={a.shape}.")
        return _nuclear_change(x)

    # 4D path (Conv feature maps)
    if a.ndim != 4:
        # Generic fallback: move channel axis to front and flatten remaining dims.
        x = np.moveaxis(a, channel_axis, 0).reshape(a.shape[channel_axis], -1)
        return _nuclear_change(x)

    # Normalize layout to NCHW so both Keras and PyTorch use identical math.
    if channel_axis != 1:
        a = np.moveaxis(a, channel_axis, 1)

    n, c, h, w = a.shape
    flat = a.reshape(n, c, h * w).transpose(1, 0, 2)  # (C, N, HW)

    if max_spatial is not None and h * w > int(max_spatial):
        idx = np.linspace(0, (h * w) - 1, num=max_spatial, dtype=np.int64)
        flat = flat[:, :, idx]
    return _nuclear_change(flat.reshape(c, -1))
