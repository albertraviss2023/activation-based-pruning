"""ReduCNN: A Dual-Framework Structural Pruning Library.

This package provides tools for structural pruning of deep learning models in both 
PyTorch and Keras/TensorFlow. It focuses on activation-based and weight-based 
pruning criteria while maintaining structural integrity for complex architectures 
like ResNet and VGG.

Main Components:
    - Core: Base adapters and common decorators.
    - Backends: Framework-specific implementations (Torch, Keras).
    - Pruner: Importance scoring, mask building, and structural surgery.
    - Analyzer: Diagnostic tools for comparing methods and Pareto frontiers.
    - Engine: High-level orchestrators for full experiments.
"""

from .core.storage import CloudStorage
from .engine.orchestrator import Orchestrator
from .pruner.surgeon import ReduCNNPruner
from .analyzer.validator import MethodValidator
from .analyzer.pareto import ParetoAnalyzer

__version__ = "0.88.0"
__all__ = [
    "CloudStorage",
    "Orchestrator",
    "ReduCNNPruner",
    "MethodValidator",
    "ParetoAnalyzer"
]
