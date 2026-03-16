from typing import Any
from ..core.decorators import get_framework_adapter

def get_adapter(model: Any, config: dict = None):
    """
    Factory function to get the appropriate adapter for a given model.
    """
    if config is None:
        config = {}
    return get_framework_adapter(model, config)
