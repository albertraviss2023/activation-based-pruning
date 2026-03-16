from typing import Dict, Any, Tuple
import numpy as np

from ..core.decorators import framework_dispatch, logger
from ..pruner.mask_builder import build_pruning_masks

class SurgicalPruner:
    """
    The core engine for performing activation-based structural pruning.
    Designed to be framework-agnostic.
    """
    
    def __init__(self, method: str = 'taylor', scope: str = 'local', config: dict = None):
        """
        Args:
            method: The pruning heuristic (e.g., 'taylor', 'l1_norm', 'apoz').
            scope: 'local' (per-layer) or 'global' (entire network).
            config: Optional dict for advanced adapter configurations.
        """
        self.method = method
        self.scope = scope
        self.config = config or {}

    @logger("Executing Surgical Pruning")
    @framework_dispatch
    def prune(self, model: Any, loader: Any, ratio: float, adapter: Any = None) -> Tuple[Any, Dict[str, np.ndarray], float]:
        """
        Analyzes the model using the provided dataloader, generates pruning masks,
        and physically rebuilds the model to be smaller.
        
        Args:
            model: PyTorch nn.Module or Keras Model.
            loader: PyTorch DataLoader or tf.data.Dataset.
            ratio: Float between 0.0 and 1.0 (e.g., 0.4 means prune 40% of filters).
            adapter: Automatically injected by @framework_dispatch.
            
        Returns:
            Tuple of (pruned_model, masks_dictionary, pruning_duration_seconds)
        """
        import time
        start_time = time.time()
        
        if adapter is None:
            raise ValueError("Adapter injection failed. Ensure @framework_dispatch is working.")
            
        print(f"🔍 Analyzing model using '{self.method}' method...")
        score_map = adapter.get_score_map(model, loader, self.method)
        
        print(f"🏗️ Building masks (scope: {self.scope}, ratio: {ratio})...")
        masks = build_pruning_masks(score_map, ratio=ratio, scope=self.scope)
        
        print("✂️ Applying physical surgery...")
        pruned_model = adapter.apply_surgery(model, masks)
        
        duration = time.time() - start_time
        return pruned_model, masks, duration
