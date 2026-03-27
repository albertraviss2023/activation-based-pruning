from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from ..core.decorators import logger
from ..pruner.surgeon import ReduCNNPruner
from ..visualization.stakeholder import (
    plot_training_history, 
    plot_layer_sensitivity, 
    plot_metrics_comparison
)

class Orchestrator:
    """Automates the 'Full Research Pipeline' for structural pruning experiments.

    The Orchestrator provides a high-level API to execute a standard sequence:
    1. Baseline Training (optional if a pre-trained model is provided).
    2. Structural Pruning (based on a specified criterion like APoZ, Mean-Activation, or L1).
    3. Fine-tuning (recovery phase).
    4. Comprehensive evaluation and visualization of the results.

    Attributes:
        config (dict): Configuration containing experimental hyperparameters.
    """
    
    def __init__(self, config: dict):
        """Initializes the Orchestrator.

        Args:
            config (dict): Configuration dictionary containing:
                - 'model_type' (str): e.g., 'vgg16' or 'resnet18'.
                - 'epochs' (int): Baseline training epochs.
                - 'ft_epochs' (int): Fine-tuning epochs.
                - 'ratio' (float): Target pruning ratio.
                - 'method' (str): Pruning criterion (e.g., 'apoz').
                - 'scope' (str): 'local' or 'global'.
        """
        self.config = config

    @logger("Starting Full Research Pipeline")
    def run(self, loader: Any, val_loader: Optional[Any] = None, 
            model: Optional[Any] = None, adapter: Any = None) -> Tuple[Any, Dict[str, Any]]:
        """Executes the end-to-end pruning and fine-tuning workflow.

        Args:
            loader (Any): Training data loader/generator.
            val_loader (Any, optional): Validation data. Defaults to None.
            model (Any, optional): Pre-trained model. If None, the baseline is 
                trained from scratch. Defaults to None.
            adapter (Any, optional): Framework adapter. If omitted, the
                adapter is resolved from `model` or `config['backend']`.

        Returns:
            Tuple[Any, Dict[str, Any]]: A tuple containing (pruned_model, masks).

        Raises:
            ValueError: If the framework adapter injection fails.
        """
        if adapter is None:
            if model is not None:
                from ..backends.factory import get_adapter
                adapter = get_adapter(model, self.config)
            else:
                backend = str(self.config.get('backend', 'pytorch')).lower().strip()
                if backend in ('pytorch', 'torch', 'pt'):
                    from ..backends.torch_backend import PyTorchAdapter
                    adapter = PyTorchAdapter(self.config)
                elif backend in ('keras', 'tensorflow', 'tf'):
                    from ..backends.keras_backend import KerasAdapter
                    adapter = KerasAdapter(self.config)
                else:
                    raise ValueError(
                        "Unable to resolve backend adapter. Provide `model`, `adapter`, "
                        "or set config['backend'] to 'pytorch' or 'keras'."
                    )

        # --- 1. Baseline Phase ---
        if model is None:
            # Train a baseline model if none was provided
            print("🚀 Training Baseline from scratch...")
            model = adapter.get_model(self.config.get('model_type', 'vgg16'))
            pretrained_path = self.config.get("pretrained_checkpoint_path")
            if pretrained_path:
                adapter.load_checkpoint(model, str(pretrained_path))
                print(f"📂 Loaded baseline checkpoint: {pretrained_path}")
            else:
                h_base = adapter.train(model, loader, self.config.get('epochs', 5), 
                                      "Baseline", val_loader=val_loader)
                plot_training_history(h_base, "Baseline Training")
        else:
            print("📂 Using provided pre-trained model...")
            pretrained_path = self.config.get("pretrained_checkpoint_path")
            if pretrained_path:
                adapter.load_checkpoint(model, str(pretrained_path))
                print(f"📂 Loaded provided-model checkpoint: {pretrained_path}")
            
        # Capture baseline statistics (FLOPs, Parameters, Accuracy)
        b_stats = adapter.get_stats(model)
        base_acc = adapter.evaluate(model, val_loader or loader)
        print(f"📊 Baseline Accuracy: {base_acc:.2f}%")

        # --- 2. Pruning Phase ---
        # Perform structural surgery based on the importance scores
        surgeon = ReduCNNPruner(
            method=self.config.get('method', 'apoz'),
            scope=self.config.get('scope', 'local'),
            config=self.config
        )
        pruned_model, masks, _ = surgeon.prune(
            model,
            loader,
            ratio=self.config.get('ratio', 0.4),
            adapter=adapter,
            save_pruned_path=self.config.get("pruned_checkpoint_path"),
        )
        
        # --- 3. Fine-tuning Phase ---
        # Recover accuracy lost during the pruning step
        print("🔥 Fine-tuning pruned model...")
        h_ft = adapter.train(pruned_model, loader, self.config.get('ft_epochs', 5), 
                            "Pruned", val_loader=val_loader)
        plot_training_history(h_ft, "Fine-tuning")

        final_checkpoint_path = self.config.get("final_checkpoint_path")
        if final_checkpoint_path:
            out = Path(str(final_checkpoint_path))
            out.parent.mkdir(parents=True, exist_ok=True)
            adapter.save_checkpoint(pruned_model, str(out))
            print(f"💾 Saved fine-tuned checkpoint: {out}")
        
        # --- 4. Evaluation & Comparison ---
        # Compare the efficiency gains against the original model
        p_stats = adapter.get_stats(pruned_model)
        final_acc = adapter.evaluate(pruned_model, val_loader or loader)
        
        print(f"\n✅ Pipeline Complete.")
        print(f"   Final Accuracy: {final_acc:.2f}% (vs {base_acc:.2f}% baseline)")
        
        # Visualize which layers were pruned most aggressively
        plot_layer_sensitivity(masks, title_prefix=self.config.get('model_type', 'Model'))
        # Generate comparative bar charts for FLOPs and Parameter count
        plot_metrics_comparison(b_stats, p_stats)
        
        return pruned_model, masks
