import numpy as np
from typing import Dict, Any

def build_pruning_masks(score_map: Dict[str, np.ndarray], ratio: float, 
                        scope: str = "local") -> Dict[str, np.ndarray]:
    """Generates binary keep-masks from channel/filter importance scores.

    Based on the provided pruning ratio and scope, this function determines
    which channels should be kept and which should be removed.

    Convention:
    - Higher score = more important = kept.
    - ratio: Pruning ratio (0.0 means keep all, 0.9 means prune 90%).
    - scope: 'local' (layer-wise pruning) or 'global' (prune across all layers).

    Args:
        score_map (Dict[str, np.ndarray]): A dictionary mapping layer names 
            to 1D arrays of filter importance scores.
        ratio (float): The target pruning ratio (0.0 <= ratio < 1.0).
        scope (str): The thresholding scope, either "local" or "global". 
            Defaults to "local".

    Returns:
        Dict[str, np.ndarray]: A mapping of layer names to boolean masks
            where True = keep and False = prune.

    Raises:
        ValueError: If ratio is outside [0, 1) or if scope is invalid.
    """
    if not (0.0 <= ratio < 1.0):
        raise ValueError("ratio must be in [0, 1).")
    if not score_map:
        return {}

    scope = scope.lower().strip()
    masks: Dict[str, np.ndarray] = {}

    if scope == "global":
        # Concatenate all scores from all layers into a single array
        all_scores = np.concatenate([np.asarray(v).reshape(-1) for v in score_map.values()]).astype("float64")
        total = all_scores.size
        # Calculate how many filters to keep across the entire model
        keep_total = max(1, int(round(total * (1.0 - ratio))))
        
        # Partition the array to find the threshold value corresponding to the top-k scores
        # np.partition is more efficient than a full sort.
        thresh = np.partition(all_scores, total - keep_total)[total - keep_total]

        for name, s in score_map.items():
            s = np.asarray(s, dtype="float64").reshape(-1)
            # Create a mask where scores >= threshold are kept
            m = s >= thresh
            
            # Critical Safety Step: Ensure we don't completely collapse a layer.
            # If a layer has no scores above the global threshold, keep its single 
            # best filter to prevent structural breakdown (e.g., zero-width layer).
            if m.sum() == 0:
                m[np.argmax(s)] = True
            masks[name] = m
            
    elif scope == "local":
        for name, s in score_map.items():
            s = np.asarray(s, dtype="float64").reshape(-1)
            # Calculate how many filters to keep for this specific layer
            keep = max(1, int(round(s.size * (1.0 - ratio))))
            
            # Identify indices of the top-k scoring filters
            idx = np.argpartition(s, -keep)[-keep:]
            
            # Create the binary mask
            m = np.zeros_like(s, dtype=bool)
            m[idx] = True
            masks[name] = m
    else:
        raise ValueError("scope must be 'local' or 'global'.")

    return masks
