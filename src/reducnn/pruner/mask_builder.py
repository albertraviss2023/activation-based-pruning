import numpy as np
from typing import Dict, Any, List, Optional

def build_pruning_masks(score_map: Dict[str, np.ndarray], ratio: float, 
                        scope: str = "local",
                        clusters: Optional[Dict[int, List[str]]] = None) -> Dict[str, np.ndarray]:
    """Generates binary keep-masks from channel/filter importance scores.

    Based on the provided pruning ratio and scope, this function determines
    which channels should be kept and which should be removed.

    Convention:
    - Higher score = more important = kept.
    - ratio: Pruning ratio (0.0 means keep all, 0.9 means prune 90%).
    - scope: 'local' (layer-wise pruning) or 'global' (prune across all layers).
    - clusters: Optional mapping of cluster IDs to layer names for mask harmonization.

    Args:
        score_map (Dict[str, np.ndarray]): A dictionary mapping layer names 
            to 1D arrays of filter importance scores.
        ratio (float): The target pruning ratio (0.0 <= ratio < 1.0).
        scope (str): The thresholding scope, either "local" or "global". 
            Defaults to "local".
        clusters (Optional[Dict[int, List[str]]]): Mapping of cluster IDs to layer names.

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

    # Pre-process score_map if clusters are provided
    # For mask harmonization, we average scores across all layers in a cluster
    # so they all get the same thresholding decision.
    effective_score_map = score_map.copy()
    if clusters:
        for c_id, members in clusters.items():
            # Find members that actually have scores
            valid_members = [m for m in members if m in score_map]
            if not valid_members: continue
            
            # Harmonization Strategy: Average scores across all layers in the cluster
            # All layers in a cluster must have the same number of filters.
            stack = np.stack([score_map[m] for m in valid_members])
            avg_scores = np.mean(stack, axis=0)
            
            for m in valid_members:
                effective_score_map[m] = avg_scores

    if scope == "global":
        # Global pruning with cluster awareness:
        # 1. Normalize scores per prunable unit (layer or cluster) 
        # to ensure comparability across different architectural depths.
        # We use L2 normalization as it's a standard literature recommendation.
        
        layer_to_cluster: Dict[str, Optional[int]] = {}
        if clusters:
            for c_id, members in clusters.items():
                for m in members:
                    layer_to_cluster[m] = c_id

        grouped_scores: Dict[Any, np.ndarray] = {}
        grouped_members: Dict[Any, List[str]] = {}
        for name, s in effective_score_map.items():
            c_id = layer_to_cluster.get(name, None)
            key = ("cluster", c_id) if c_id is not None else ("layer", name)
            if key not in grouped_scores:
                # Use L2 norm for inter-layer comparability
                s_arr = np.asarray(s, dtype="float64").reshape(-1)
                norm = np.sqrt(np.sum(np.square(s_arr))) + 1e-12
                grouped_scores[key] = s_arr / norm
                grouped_members[key] = []
            grouped_members[key].append(name)

        # Use exact top-k selection globally to avoid tie-related over-keeping.
        # Threshold-based selection (>=) can keep everything when many values tie.
        group_items = list(grouped_scores.items())
        flat_scores = np.concatenate([v.reshape(-1) for _, v in group_items]).astype("float64")
        total = flat_scores.size
        keep_total = max(1, int(round(total * (1.0 - ratio))))

        selected = np.zeros(total, dtype=bool)
        if keep_total >= total:
            selected[:] = True
        else:
            topk_idx = np.argpartition(flat_scores, -keep_total)[-keep_total:]
            selected[topk_idx] = True

        group_masks: Dict[Any, np.ndarray] = {}
        cursor = 0
        for key, s in group_items:
            n = s.size
            m = selected[cursor:cursor + n].copy()
            if m.sum() == 0 and n > 0:
                m[np.argmax(s)] = True
            group_masks[key] = m
            cursor += n

        for key, members in grouped_members.items():
            gm = group_masks[key]
            for name in members:
                masks[name] = gm.copy()
            
    elif scope == "local":
        for name, s in effective_score_map.items():
            s = np.asarray(s, dtype="float64").reshape(-1)
            keep = max(1, int(round(s.size * (1.0 - ratio))))
            idx = np.argpartition(s, -keep)[-keep:]
            
            m = np.zeros_like(s, dtype=bool)
            m[idx] = True
            masks[name] = m
    else:
        raise ValueError("scope must be 'local' or 'global'.")

    return masks
