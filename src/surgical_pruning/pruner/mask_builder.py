import numpy as np
from typing import Dict

def build_pruning_masks(score_map: Dict[str, np.ndarray], ratio: float, scope: str = "local") -> Dict[str, np.ndarray]:
    """
    Build boolean keep-masks from per-filter importance scores.

    Convention:
    - Higher score = more important = keep.
    - ratio is prune ratio (0.0 to 1.0).
    - scope:
      - local: threshold independently per layer
      - global: threshold across all layers together
    """
    if not (0.0 <= ratio < 1.0):
        raise ValueError("ratio must be in [0, 1).")
    if not score_map:
        return {}

    scope = scope.lower().strip()
    masks: Dict[str, np.ndarray] = {}

    if scope == "global":
        all_scores = np.concatenate([np.asarray(v).reshape(-1) for v in score_map.values()]).astype("float64")
        total = all_scores.size
        keep_total = max(1, int(round(total * (1.0 - ratio))))
        thresh = np.partition(all_scores, total - keep_total)[total - keep_total]

        for name, s in score_map.items():
            s = np.asarray(s, dtype="float64").reshape(-1)
            m = s >= thresh
            # Ensure we don't completely collapse a layer
            if m.sum() == 0:
                m[np.argmax(s)] = True
            masks[name] = m
            
    elif scope == "local":
        for name, s in score_map.items():
            s = np.asarray(s, dtype="float64").reshape(-1)
            keep = max(1, int(round(s.size * (1.0 - ratio))))
            idx = np.argpartition(s, -keep)[-keep:]
            m = np.zeros_like(s, dtype=bool)
            m[idx] = True
            masks[name] = m
    else:
        raise ValueError("scope must be 'local' or 'global'.")

    return masks
