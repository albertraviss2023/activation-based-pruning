from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional
import numpy as np

class FrameworkAdapter(ABC):
    """
    Abstract Base Class (ABC) defining the standard interface for Deep Learning frameworks.
    
    This adapter pattern allows the ReduCNN engine to remain framework-agnostic. 
    Every backend (PyTorch, Keras, etc.) must implement these methods to ensure 
    compatibility with the global pruning orchestrator.
    
    Attributes:
        config (Dict[str, Any]): Configuration dictionary containing training 
            hyperparameters, device settings, and model metadata.
    """

    @abstractmethod
    def get_model(self, 
                  model_type: str, 
                  input_shape: Optional[Tuple[int, ...]] = None, 
                  num_classes: Optional[int] = None) -> Any:
        """
        Initializes and returns a standard architecture model.
        
        Args:
            model_type (str): Identifier for the architecture (e.g., 'vgg16', 'resnet18').
            input_shape (Optional[Tuple[int, ...]]): The expected input tensor shape 
                (e.g., (3, 32, 32) for Torch or (32, 32, 3) for Keras).
            num_classes (Optional[int]): Number of output units for the final layer.
            
        Returns:
            Any: A framework-specific model object (nn.Module or tf.keras.Model).
            
        Raises:
            ValueError: If the model_type is not recognized.
        """
        pass

    @abstractmethod
    def train(self, 
              model: Any, 
              loader: Any, 
              epochs: int, 
              name: str, 
              val_loader: Optional[Any] = None, 
              plot: bool = True) -> Dict[str, List[float]]:
        """
        Standardizes the training loop across frameworks.
        
        Args:
            model (Any): The model to be trained.
            loader (Any): Training data iterator/loader.
            epochs (int): Number of full passes over the dataset.
            name (str): Unique name for this training run (used for logging/checkpoints).
            val_loader (Optional[Any]): Validation data iterator.
            plot (bool): If True, renders a training history chart upon completion.
            
        Returns:
            Dict[str, List[float]]: History of loss and accuracy metrics for train/val.
        """
        pass

    @abstractmethod
    def evaluate(self, model: Any, loader: Any) -> float:
        """
        Calculates the top-1 accuracy of the model on a dataset.
        
        Args:
            model (Any): The model to evaluate.
            loader (Any): Dataset iterator.
            
        Returns:
            float: Accuracy percentage (0.0 to 100.0).
        """
        pass

    @abstractmethod
    def get_score_map(self, model: Any, loader: Any, method: str) -> Dict[str, np.ndarray]:
        """
        Calculates importance scores for every filter in all target layers.
        
        Args:
            model (Any): The model to analyze.
            loader (Any): Representative data used for data-dependent heuristics.
            method (str): The pruning heuristic to use (e.g., 'l1_norm', 'taylor').
            
        Returns:
            Dict[str, np.ndarray]: A mapping of layer names to 1D arrays of importance scores.
        """
        pass

    @abstractmethod
    def apply_surgery(self, model: Any, masks: Dict[str, np.ndarray]) -> Any:
        """
        Performs physical structural surgery by deleting channels from the model.
        
        Args:
            model (Any): The original "fat" model.
            masks (Dict[str, np.ndarray]): Binary masks (1=keep, 0=prune) for each layer.
            
        Returns:
            Any: A new, physically smaller model with reduced parameters and FLOPs.
        """
        pass

    @abstractmethod
    def get_stats(self, model: Any, loader: Optional[Any] = None) -> Tuple[float, float]:
        """
        Calculates the computational footprint of the model.
        
        Args:
            model (Any): The model to profile.
            loader (Optional[Any]): Data loader to derive dynamic input shapes.
            
        Returns:
            Tuple[float, float]: (FLOPs, Total Parameters).
        """
        pass

    @abstractmethod
    def save_checkpoint(self, model: Any, path: str) -> None:
        """Saves model weights to disk."""
        pass

    @abstractmethod
    def load_checkpoint(self, model: Any, path: str) -> None:
        """Loads model weights from disk."""
        pass

    @abstractmethod
    def trace_graph(self, model: Any) -> Dict[str, Any]:
        """
        Traces the model's computational graph to identify dependencies.
        
        Returns:
            Dict[str, Any]: A standardized graph representation containing
                nodes, edges, and structural clusters.
        """
        pass

    @abstractmethod
    def classify_architecture(self, model: Any) -> str:
        """
        Categorizes the model based on its topological features.
        
        Returns:
            str: 'sequential', 'residual', or 'concatenative'.
        """
        pass

    @abstractmethod
    def get_multi_metric_scores(self, model: Any, loader: Any, metrics: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Calculates multiple pruning metrics in a single optimization pass.
        
        Returns:
            Dict[str, Dict[str, np.ndarray]]: Mapping of metric_name -> {layer_name: scores}.
        """
        pass
