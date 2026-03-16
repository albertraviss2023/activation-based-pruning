import numpy as np
from typing import Dict, Any, List
from ..core.decorators import framework_dispatch, logger
from ..pruner.surgeon import SurgicalPruner
from ..visualization.research import plot_rank_correlation, plot_score_distributions

class MethodValidator:
    """
    Diagnostic tool to compare different pruning methods (e.g., L1 vs Taylor vs APoZ)
    without permanently altering the model.
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        
    @logger("Running Method Validation Suite")
    @framework_dispatch
    def compare_methods(self, model: Any, loader: Any, methods: List[str], ratio: float = 0.3, adapter: Any = None):
        """
        Runs multiple pruning criteria on the same model and plots their correlations.
        
        Args:
            model: The neural network.
            loader: The dataset.
            methods: List of method strings (e.g., ['l1_norm', 'taylor', 'random']).
            ratio: Pruning ratio to simulate.
            adapter: Auto-injected.
        """
        if adapter is None:
            raise ValueError("Adapter injection failed.")
            
        score_maps = {}
        for method in methods:
            print(f"Calculating scores for: {method}...")
            # We bypass the SurgicalPruner to just get the raw scores
            score_maps[method] = adapter.get_score_map(model, loader, method)
            
        print("\nGenerating Diagnostic Plots...")
        plot_score_distributions(score_maps, title_prefix="Method Validation")
        plot_rank_correlation(score_maps, title_prefix="Method Validation")
        print("Validation suite complete.")
