import matplotlib.pyplot as plt
from typing import Any, List, Tuple, Optional
from ..core.decorators import framework_dispatch, logger

class ParetoAnalyzer:
    """Analyzes the trade-off between model efficiency and predictive accuracy.

    This tool iteratively prunes a model at various ratios to determine the 
    Pareto Frontier—the set of pruning configurations that represent the best 
    possible accuracy for a given computational budget (FLOPs).

    Attributes:
        method (str): The pruning criterion to test (e.g., 'apoz').
        scope (str): Pruning scope ('local' or 'global').
        config (dict): Configuration for the pruner.
    """
    def __init__(self, method: str = 'apoz', scope: str = 'local', 
                 config: dict = None):
        """Initializes the ParetoAnalyzer.

        Args:
            method (str): Scoring method. Defaults to 'apoz'.
            scope (str): Thresholding scope. Defaults to 'local'.
            config (dict, optional): Configuration parameters. Defaults to None.
        """
        self.method = method
        self.scope = scope
        self.config = config or {}
        
    @logger("Generating Pareto Frontier")
    @framework_dispatch
    def run(self, model: Any, loader: Any, val_loader: Optional[Any] = None, 
            ratios: List[float] = [0.2, 0.4, 0.6, 0.8], adapter: Any = None) -> None:
        """Evaluates model performance across multiple pruning ratios.

        Calculates both accuracy and FLOPs for each ratio to build the 
        trade-off analysis.

        Args:
            model (Any): A pre-trained model. Must be trained for meaningful scores.
            loader (Any): Data used for calculating importance scores.
            val_loader (Any, optional): Evaluation data. Defaults to loader if None.
            ratios (List[float]): Pruning ratios to test.
            adapter (Any, optional): Injected framework adapter.

        Raises:
            ValueError: If the framework adapter injection fails.
        """
        if adapter is None:
            raise ValueError("Adapter injection failed.")
            
        print("📊 Establishing Baseline...")
        # Get baseline computational stats and accuracy
        b_stats = adapter.get_stats(model)
        base_acc = adapter.evaluate(model, val_loader if val_loader else loader)
        print(f"   Baseline Acc: {base_acc:.2f}%, FLOPs: {b_stats[0]/1e6:.2f}M")

        # Initialize the pruner with the specified criterion
        from ..pruner.surgeon import ReduCNNPruner
        pruner = ReduCNNPruner(method=self.method, scope=self.scope, config=self.config)

        results = []
        
        # Iterate through ratios and perform structural pruning
        for r in ratios:
            print(f"\n--- Testing Prune Ratio: {r} ---")
            # Prune a copy of the model to keep the original baseline intact
            pruned_model, _, _ = pruner.prune(model, loader, ratio=r, adapter=adapter)
            
            # Evaluate the smaller model
            acc = adapter.evaluate(pruned_model, val_loader if val_loader else loader)
            p_stats = adapter.get_stats(pruned_model)
            
            # Store (ratio, accuracy, FLOPs)
            results.append((r, acc, p_stats[0]))
            print(f"   Result -> Acc: {acc:.2f}%, FLOPs: {p_stats[0]/1e6:.2f}M")
            
        # Visualize the results
        self._plot(base_acc, b_stats[0], results)
        
    def _plot(self, base_acc: float, base_flops: float, 
              results: List[Tuple[float, float, float]]) -> None:
        """Generates plots for Accuracy vs. Pruning Ratio and Accuracy vs. Compute.

        Args:
            base_acc (float): Baseline model accuracy.
            base_flops (float): Baseline model FLOPs.
            results (List[Tuple]): List of (ratio, accuracy, flops) for pruned models.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Subplot 1: Accuracy vs. Pruning Ratio
        # Shows how quickly accuracy degrades as we remove more parameters.
        x_r = [0] + [r for r, _, _ in results]
        y_a = [base_acc] + [a for _, a, _ in results]
        ax1.plot(x_r, y_a, marker='o', linestyle='-', color='b', linewidth=2)
        ax1.set_title("Accuracy vs. Pruning Ratio", fontsize=13, fontweight='bold')
        ax1.set_xlabel("Pruning Ratio (Higher = More Pruned)")
        ax1.set_ylabel("Top-1 Accuracy (%)")
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Accuracy vs. Compute (Pareto Frontier)
        # Shows the actual efficiency gain relative to the accuracy cost.
        x_f = [base_flops/1e6] + [f/1e6 for _, _, f in results]
        ax2.plot(x_f, y_a, marker='s', linestyle='-', color='g', linewidth=2)
        ax2.set_title("Pareto Frontier: Accuracy vs. Compute", fontsize=13, fontweight='bold')
        ax2.set_xlabel("Compute Cost (MFLOPs)")
        ax2.set_ylabel("Top-1 Accuracy (%)")
        # Invert X-axis so that the 'best' direction (lower FLOPs, higher Acc) 
        # is towards the top-right or follows a standard Pareto convention.
        ax2.invert_xaxis()  
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f"Pruning Trade-off Analysis ({self.method.upper()})", 
                     fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Internal test to ensure plotting logic works with dummy data
    analyzer = ParetoAnalyzer()
    dummy_results = [(0.2, 95.0, 80e6), (0.4, 92.0, 60e6), 
                     (0.6, 85.0, 40e6), (0.8, 60.0, 20e6)]
    analyzer._plot(98.0, 100e6, dummy_results)
