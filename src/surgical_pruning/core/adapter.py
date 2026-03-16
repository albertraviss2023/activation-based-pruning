import abc
from typing import Any, Dict, Callable, Tuple
import numpy as np

class FrameworkAdapter(abc.ABC):
    """
    Abstract Base Class ensuring a consistent interface across
    different deep learning frameworks (PyTorch, Keras).
    """

    @abc.abstractmethod
    def get_model(self, model_type: str) -> Any:
        """Returns a compiled/initialized model of the specified type."""
        pass

    @abc.abstractmethod
    def train(self, model: Any, loader: Any, epochs: int, name: str, val_loader: Any = None) -> Dict[str, list]:
        """Trains the model and returns a history dictionary."""
        pass

    @abc.abstractmethod
    def evaluate(self, model: Any, loader: Any) -> float:
        """Evaluates the model and returns the Top-1 Accuracy percentage."""
        pass

    @abc.abstractmethod
    def get_viz_data(self, model: Any, loader: Any, num_layers: int = 3) -> Dict[str, np.ndarray]:
        """Extracts feature maps and first-layer weights for visualization."""
        pass

    @abc.abstractmethod
    def get_stats(self, model: Any) -> Tuple[float, float]:
        """Returns (FLOPs, Parameters) for the model."""
        pass

    @abc.abstractmethod
    def save_checkpoint(self, model: Any, path: str):
        """Saves the model's weights to the specified path."""
        pass

    @abc.abstractmethod
    def load_checkpoint(self, model: Any, path: str):
        """Loads the model's weights from the specified path."""
        pass

    @abc.abstractmethod
    def get_score_map(self, model: Any, loader: Any, method: str) -> Dict[str, np.ndarray]:
        """Calculates and returns the pruning importance scores for each filter/channel."""
        pass

    @abc.abstractmethod
    def apply_surgery(self, model: Any, masks: Dict[str, np.ndarray]) -> Any:
        """Physically rebuilds the model applying the provided boolean masks."""
        pass
