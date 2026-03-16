import matplotlib.pyplot as plt
from typing import Any, List
from ..core.decorators import framework_dispatch, logger
from ..pruner.surgeon import SurgicalPruner

class ParetoAnalyzer:
    """
    Analyzes the trade-off between Efficiency (FLOPs/Pruning Ratio) and Accuracy.
    Can be run independently of the main training/pruning pipeline.
    """
    def __init__(self, method: str = 'taylor', scope: str = 'local', config: dict = None):
        self.method = method
        self.scope = scope
        self.config = config or {}
        
    @logger("Generating Pareto Frontier")
    @framework_dispatch
    def run(self, model: Any, loader: Any, val_loader: Any = None, ratios: List[float] = [0.2, 0.4, 0.6, 0.8], adapter: Any = None):
        """
        Iteratively prunes the model at various ratios and plots the accuracy drop-off.
        
        Args:
            model: A PRE-TRAINED PyTorch or Keras model. (Must be trained for meaningful results).
            loader: The dataset used for calculating pruning criteria (e.g. Taylor/Activation).
            val_loader: The dataset used for evaluating accuracy. (If None, falls back to loader).
            ratios: List of pruning ratios to test.
            adapter: Automatically injected.
        """
        if adapter is None:
            raise ValueError("Adapter injection failed.")
            
        print("📊 Establishing Baseline...")
        b_stats = adapter.get_stats(model)
        base_acc = adapter.evaluate(model, val_loader if val_loader else loader)
        print(f"   Baseline Acc: {base_acc:.2f}%, FLOPs: {b_stats[0]/1e6:.2f}M")
        
        pruner = SurgicalPruner(method=self.method, scope=self.scope, config=self.config)
        results = []
        
        for r in ratios:
            print(f"\n--- Testing Prune Ratio: {r} ---")
            # Prune a fresh copy (adapter ensures model is not mutated in-place where applicable)
            pruned_model, _ = pruner.prune(model, loader, ratio=r)
            acc = adapter.evaluate(pruned_model, val_loader if val_loader else loader)
            p_stats = adapter.get_stats(pruned_model)
            results.append((r, acc, p_stats[0]))
            print(f"   Result -> Acc: {acc:.2f}%, FLOPs: {p_stats[0]/1e6:.2f}M")
            
        self._plot(base_acc, b_stats[0], results)
        
    def _plot(self, base_acc, base_flops, results):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Accuracy vs Ratio
        x_r = [0] + [r for r, _, _ in results]
        y_a = [base_acc] + [a for _, a, _ in results]
        ax1.plot(x_r, y_a, marker='o', linestyle='-', color='b', linewidth=2)
        ax1.set_title("Accuracy vs. Pruning Ratio", fontsize=13, fontweight='bold')
        ax1.set_xlabel("Pruning Ratio (Higher = More Pruned)")
        ax1.set_ylabel("Top-1 Accuracy (%)")
        ax1.grid(True, alpha=0.3)
        
        # Accuracy vs Compute
        x_f = [base_flops/1e6] + [f/1e6 for _, _, f in results]
        ax2.plot(x_f, y_a, marker='s', linestyle='-', color='g', linewidth=2)
        ax2.set_title("Pareto Frontier: Accuracy vs. Compute", fontsize=13, fontweight='bold')
        ax2.set_xlabel("Compute Cost (MFLOPs)")
        ax2.set_ylabel("Top-1 Accuracy (%)")
        ax2.invert_xaxis()  # Lower FLOPs (right) is better efficiency
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f"Pruning Trade-off Analysis ({self.method.upper()})", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Test plotting independently
    analyzer = ParetoAnalyzer()
    dummy_results = [(0.2, 95.0, 80e6), (0.4, 92.0, 60e6), (0.6, 85.0, 40e6), (0.8, 60.0, 20e6)]
    analyzer._plot(98.0, 100e6, dummy_results)
