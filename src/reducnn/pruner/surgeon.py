import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
from ..core.decorators import timer, logger, framework_dispatch
from ..core.adapter import FrameworkAdapter
from .mask_builder import build_pruning_masks

class ReduCNNPruner:
    """
    The High-Level Pruning Orchestrator (Surgeon).
    
    This class serves as the primary interface for users to perform structural 
    channel pruning on their models. It coordinates the analysis of filter 
    importance, the selection of which filters to keep (masking), and the 
    actual physical removal of weights (surgery).
    
    Example:
        >>> surgeon = ReduCNNPruner(method='l1_norm', scope='local')
        >>> pruned_model, masks, duration = surgeon.prune(model, dataloader, ratio=0.5)
    
    Attributes:
        method (str): The importance heuristic to use (e.g., 'l1_norm', 'taylor', 'chip').
        scope (str): Pruning scope - 'local' (layer-wise) or 'global' (network-wide).
        config (Dict[str, Any]): Dictionary of framework-specific settings.
    """

    def __init__(self, method: str = 'l1_norm', scope: str = 'local', config: Optional[Dict[str, Any]] = None):
        """
        Initializes the Surgeon with a specific pruning strategy.
        
        Args:
            method (str): Metric for ranking filters. 
                Options: 'l1_norm', 'l2_norm', 'taylor', 'apoz', 'chip'.
            scope (str): How thresholds are calculated. 
                'local': Prunes the same % from every layer.
                'global': Prunes the least important filters across the whole model.
            config (Optional[Dict[str, Any]]): Configuration for backend adapters.
        """
        self.method = method.lower().strip()
        self.scope = scope.lower().strip()
        self.config = config or {}

    @framework_dispatch
    @timer
    @logger("Executing ReduCNN")
    def prune(self, 
              model: Any, 
              loader: Any, 
              ratio: float = 0.5, 
              adapter: Optional[FrameworkAdapter] = None) -> Tuple[Any, Dict[str, np.ndarray], float]:
        """
        Performs structural surgery on the model to reduce its size and complexity.
        
        This method executes a 3-step surgical pipeline:
        1. **Analysis**: Uses the chosen 'method' to score every filter's importance.
        2. **Masking**: Ranks scores and identifies filters below the 'ratio' threshold.
        3. **Surgery**: Modifies the model's internal graph to physically remove channels.
        
        Args:
            model (Any): The PyTorch (nn.Module) or Keras (tf.keras.Model) to be pruned.
            loader (Any): Data loader used for data-dependent heuristics (like Taylor/CHIP).
            ratio (float): The percentage of filters to REMOVE (0.0 to 1.0).
            adapter (Optional[FrameworkAdapter]): The backend adapter automatically 
                injected by the @framework_dispatch decorator.
            
        Returns:
            Tuple[Any, Dict[str, np.ndarray], float]: 
                - pruned_model: The new, physically smaller model.
                - masks: Binary keep/prune masks for every target layer.
                - duration: The total time (seconds) taken for the surgery phase.
                
        Raises:
            SurgeryError: If the structural graph cannot be safely reconstructed.
            ValueError: If the pruning method returns invalid or null scores.
        """
        # PHASE 1: SCORE CALCULATION
        # We delegate the math to the backend adapter which handles framework-specific
        # operations (like Torch Hooks or Keras GradientTapes).
        print(f"🔍 Analyzing model using '{self.method}' method...")
        score_map = adapter.get_score_map(model, loader, self.method)

        # PHASE 2: MASK BUILDING
        # Converts raw importance scores into binary decisions. 
        # If scope='global', we flatten all scores before thresholding.
        print(f"🏗️ Building masks (scope: {self.scope}, ratio: {ratio})...")
        masks = build_pruning_masks(score_map, ratio, scope=self.scope)

        # PHASE 3: PHYSICAL SURGERY
        # This is the most complex step where we 'shrink' the tensors and 
        # reconstruct the model graph to maintain connectivity.
        print(f"✂️ Applying physical surgery...")
        start_time = time.time()
        
        # The adapter performs framework-specific surgery (e.g., param slicing or functional rebuild)
        pruned_model = adapter.apply_surgery(model, masks)
        
        duration = time.time() - start_time

        return pruned_model, masks, duration
