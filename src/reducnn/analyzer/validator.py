import numpy as np
from typing import Dict, Any, List, Optional
from ..core.adapter import FrameworkAdapter
from ..core.decorators import framework_dispatch, logger
from ..visualization.research import (
    plot_rank_correlation,
    plot_score_distributions,
    plot_decision_agreement,
)

class MethodValidator:
    """Diagnostic tool to compare different pruning methods without model alteration.

    This class enables researchers to analyze how various pruning criteria 
    (e.g., L1-norm vs. Taylor-1 vs. APoZ) correlate with each other on a 
    specific model and dataset. It helps in identifying the most stable 
    or aggressive importance metrics before committing to a structural prune.

    Attributes:
        config (dict): Configuration parameters for the validation process.
    """
    
    def __init__(self, config: dict = None):
        """Initializes the MethodValidator.

        Args:
            config (dict, optional): Configuration dictionary. Defaults to None.
        """
        self.config = config or {}
        
    @logger("Running Method Validation Suite")
    @framework_dispatch
    def compare_methods(self, model: Any, loader: Any, methods: List[str], 
                        ratio: float = 0.3, adapter: Any = None) -> None:
        """Calculates multiple pruning criteria and visualizes their relationships.

        Runs the specified pruning methods on the same model and generates 
        distribution and correlation plots.

        Args:
            model (Any): The neural network model (PyTorch or Keras).
            loader (Any): The dataset/data loader used for scoring (calibration).
            methods (List[str]): List of pruning method names to compare 
                (e.g., ['l1_norm', 'taylor', 'random', 'apoz']).
            ratio (float, optional): Pruning ratio to simulate in visualizations. 
                Defaults to 0.3.
            adapter (Any, optional): The framework adapter, automatically 
                injected by @framework_dispatch.

        Raises:
            ValueError: If the framework adapter injection fails.
        """
        if adapter is None:
            raise ValueError("Adapter injection failed. Ensure @framework_dispatch is working correctly.")
            
        score_maps = {}
        # Iterate through each requested method and extract raw channel scores
        for method in methods:
            print(f"Calculating scores for: {method}...")
            # We bypass the full ReduCNNPruner pipeline and call the adapter's
            # score mapping logic directly to avoid unnecessary model copies.
            score_maps[method] = adapter.get_score_map(model, loader, method)
            
        print("\nGenerating Diagnostic Plots...")
        # Plot the probability density of scores for each method
        plot_score_distributions(score_maps, title_prefix="Method Validation")
        # Plot Spearman/Pearson rank correlations between different methods
        plot_rank_correlation(score_maps, title_prefix="Method Validation")
        # Plot top-k decision agreement (what would actually be kept/pruned)
        plot_decision_agreement(score_maps, ratio=ratio, title_prefix="Method Validation")
        
        print("Validation suite complete.")

class ModelValidator:
    """Validator to ensure custom models are compatible with ReduCNN.
    
    Checks if a model can be successfully traced and if it contains 
    layers that are prunable (e.g., Conv2D).
    """
    
    def validate_model(self, model: Any, adapter: FrameworkAdapter) -> bool:
        """Runs a suite of compatibility checks on the model.
        
        Args:
            model (Any): The model instance to validate.
            adapter (FrameworkAdapter): The framework-specific adapter.
            
        Returns:
            bool: True if the model is compatible, False otherwise.
        """
        try:
            # 1. Check for traceability
            graph = adapter.trace_graph(model)
            if not graph or "nodes" not in graph:
                return False
            
            # 2. Check for prunable layers
            prunable_nodes = [n for n, d in graph["nodes"].items() if d.get("type") == "conv2d"]
            if not prunable_nodes:
                print("⚠️ Warning: No prunable Conv2D layers detected in the model.")
                return False
                
            return True
        except Exception as e:
            print(f"❌ Model validation error: {e}")
            return False
