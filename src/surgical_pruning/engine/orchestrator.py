from typing import Any, Dict, Optional
from ..core.decorators import framework_dispatch, logger
from ..pruner.surgeon import SurgicalPruner
from ..visualization.stakeholder import (
    plot_training_history, 
    plot_layer_sensitivity, 
    plot_metrics_comparison
)

class Orchestrator:
    """
    A high-level convenience class for the 'Full Research Pipeline'.
    
    This class automates the Train -> Prune -> Fine-tune -> Compare sequence.
    It remains decoupled by delegating all logic to specific modules.
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dictionary containing:
                - 'model_type': e.g. 'vgg16'
                - 'epochs': Baseline training epochs
                - 'ft_epochs': Fine-tuning epochs
                - 'ratio': Pruning ratio
                - 'method': Pruning criteria
        """
        self.config = config

    @logger("Starting Full Research Pipeline")
    @framework_dispatch
    def run(self, loader: Any, val_loader: Optional[Any] = None, model: Optional[Any] = None, adapter: Any = None):
        """
        Executes the end-to-end workflow.
        
        If `model` is provided, it skips baseline training and prunes the provided model.
        """
        if adapter is None:
            raise ValueError("Adapter injection failed.")

        # 1. Baseline Phase
        if model is None:
            print("🚀 Training Baseline from scratch...")
            model = adapter.get_model(self.config.get('model_type', 'vgg16'))
            h_base = adapter.train(model, loader, self.config.get('epochs', 5), "Baseline", val_loader=val_loader)
            plot_training_history(h_base, "Baseline Training")
        else:
            print("📂 Using provided pre-trained model...")
            
        b_stats = adapter.get_stats(model)
        base_acc = adapter.evaluate(model, val_loader or loader)
        print(f"📊 Baseline Accuracy: {base_acc:.2f}%")

        # 2. Pruning Phase
        surgeon = SurgicalPruner(
            method=self.config.get('method', 'taylor'),
            scope=self.config.get('scope', 'local'),
            config=self.config
        )
        pruned_model, masks = surgeon.prune(model, loader, ratio=self.config.get('ratio', 0.4))
        
        # 3. Fine-tuning Phase
        print("🔥 Fine-tuning pruned model...")
        h_ft = adapter.train(pruned_model, loader, self.config.get('ft_epochs', 5), "Pruned", val_loader=val_loader)
        plot_training_history(h_ft, "Fine-tuning")
        
        # 4. Evaluation & Comparison
        p_stats = adapter.get_stats(pruned_model)
        final_acc = adapter.evaluate(pruned_model, val_loader or loader)
        
        print(f"\n✅ Pipeline Complete.")
        print(f"   Final Accuracy: {final_acc:.2f}% (vs {base_acc:.2f}% baseline)")
        
        plot_layer_sensitivity(masks, title_prefix=self.config.get('model_type', 'Model'))
        plot_metrics_comparison(b_stats, p_stats)
        
        return pruned_model, masks
