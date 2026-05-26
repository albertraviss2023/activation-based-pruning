import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
from pathlib import Path
from ..core.decorators import timer, logger, framework_dispatch
from ..core.adapter import FrameworkAdapter
from .mask_builder import build_pruning_masks

from ..analyzer.classifier import ArchitectureClassifier
from ..analyzer.validator import ModelValidator

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
        method (str): The importance heuristic to use (e.g., 'l1_norm', 'mean_abs_act', 'apoz').
        scope (str): Pruning scope - 'local' (layer-wise) or 'global' (network-wide).
        config (Dict[str, Any]): Dictionary of framework-specific settings.
    """

    def __init__(self, method: str = 'l1_norm', scope: str = 'local', config: Optional[Dict[str, Any]] = None):
        """
        Initializes the Surgeon with a specific pruning strategy.
        
        Args:
            method (str): Metric for ranking filters.
                Bundled options: 'l1_norm', 'mean_abs_act', 'apoz'.
                Additional methods can be added via `@register_method(...)`.
            scope (str): How thresholds are calculated. 
                'local': Prunes the same % from every layer.
                'global': Prunes the least important filters across the whole model.
            config (Optional[Dict[str, Any]]): Configuration for backend adapters.
        """
        self.method = method.lower().strip()
        self.scope = scope.lower().strip()
        self.config = config or {}
        self.last_timing: Dict[str, float] = {}

    @framework_dispatch
    @timer
    @logger("Executing ReduCNN")
    def prune(self, 
              model: Any, 
              loader: Any, 
              ratio: float = 0.5, 
              adapter: Optional[FrameworkAdapter] = None,
              save_pruned_path: Optional[str] = None) -> Tuple[Any, Dict[str, np.ndarray], float]:
        """
        Performs structural surgery on the model to reduce its size and complexity.
        """
        pipeline_start_time = time.time()
        self.last_timing = {}

        # PHASE 0: TOPOLOGY ANALYSIS
        print(f"🌐 Analyzing model topology...")
        topology_start_time = time.time()
        classifier = ArchitectureClassifier(adapter)
        clusters = classifier.get_clusters(model)
        topo_type = classifier.get_topology_type(model)
        topology_time = time.time() - topology_start_time
        print(f"✅ Detected {topo_type} architecture with {len(clusters)} pruning clusters.")

        # PHASE 1: SCORE CALCULATION
        self.config["ratio"] = ratio
        self.config["current_prune_ratio"] = ratio
        if adapter is not None and hasattr(adapter, "config") and isinstance(adapter.config, dict):
            adapter.config.setdefault("ratio", ratio)
            adapter.config["current_prune_ratio"] = ratio

        score_start_time = time.time()
        if self.method in ('hybrid', 'meta'):
            print(f"🧠 Executing Hybrid Meta-Pruning Engine (Literature-Grounded)...")
            from .meta_criteria import HybridMetaPruner
            meta_engine = HybridMetaPruner(adapter, mode=self.config.get('meta_mode', 'smooth'))
            score_map = meta_engine.calculate_hybrid_scores(model, loader)
        else:
            print(f"🔍 Analyzing model using '{self.method}' method...")
            score_map = adapter.get_score_map(model, loader, self.method)
        score_time = time.time() - score_start_time

        # PHASE 2: MASK BUILDING
        print(f"🏗️ Building masks (scope: {self.scope}, ratio: {ratio})...")
        mask_start_time = time.time()
        masks = build_pruning_masks(score_map, ratio, scope=self.scope, clusters=clusters)
        mask_build_time = time.time() - mask_start_time

        # PHASE 3: PHYSICAL SURGERY
        print(f"✂️ Applying physical surgery...")
        surgery_start_time = time.time()
        pruned_model = adapter.apply_surgery(model, masks)
        surgery_time = time.time() - surgery_start_time

        if save_pruned_path:
            out = Path(str(save_pruned_path))
            out.parent.mkdir(parents=True, exist_ok=True)
            adapter.save_checkpoint(pruned_model, save_pruned_path)
            print(f"💾 Saved pruned checkpoint to: {out}")

        pipeline_time = time.time() - pipeline_start_time
        method_cost_time = score_time + mask_build_time + surgery_time
        self.last_timing = {
            "topology_time_sec": float(topology_time),
            "score_time_sec": float(score_time),
            "mask_build_time_sec": float(mask_build_time),
            "surgery_time_sec": float(surgery_time),
            "method_cost_time_sec": float(method_cost_time),
            "prune_pipeline_time_sec": float(pipeline_time),
        }

        return pruned_model, masks, method_cost_time

    @framework_dispatch
    def prune_custom_model(self, model: Any, loader: Any, ratio: float = 0.5,
                           adapter: Optional[FrameworkAdapter] = None,
                           checkpoint_path: Optional[str] = None,
                           save_pruned_path: Optional[str] = None) -> Tuple[Any, Dict[str, np.ndarray], float]:
        """
        Unified API endpoint for pruning user-provided custom pre-trained models.
        
        This method skips the baseline training phase and performs direct 
        sensitivity analysis and pruning.
        
        Args:
            model (Any): Pre-trained framework-specific model.
            loader (Any): Calibration data.
            ratio (float): Target pruning ratio.
            adapter (Optional[FrameworkAdapter]): Automatically injected.
            
        Returns:
            Tuple[Any, Dict[str, np.ndarray], float]: Pruned model and metadata.
        """
        print("📥 Importing custom pre-trained model...")
        validator = ModelValidator()
        if not validator.validate_model(model, adapter):
            raise ValueError("Model validation failed. Ensure the model is traceable and prunable.")

        if checkpoint_path:
            adapter.load_checkpoint(model, checkpoint_path)
            print(f"📂 Loaded pre-trained checkpoint from: {checkpoint_path}")

        return self.prune(
            model,
            loader,
            ratio=ratio,
            adapter=adapter,
            save_pruned_path=save_pruned_path,
        )
