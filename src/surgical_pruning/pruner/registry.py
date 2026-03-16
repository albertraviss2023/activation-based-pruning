import inspect
from typing import Callable, Dict, Any
from ..core.exceptions import MethodRegistrationError

_PRUNING_REGISTRY: Dict[str, Callable[..., Any]] = {}

def register_method(name: str):
    """
    Decorator to register a custom pruning score function.
    
    The registered function should accept at minimum:
        - `layer`: The Conv2D layer (PyTorch or Keras)
        - `model`: The full model
    And return a 1D numpy array or tensor of scores (length = output channels).
    Higher score = keep filter.
    """
    def decorator(func: Callable):
        if not callable(func):
            raise MethodRegistrationError(f"Pruning method {name} must be callable.")
        _PRUNING_REGISTRY[name.lower()] = func
        return func
    return decorator

def get_method(name: str) -> Callable:
    """Retrieves a pruning method from the registry."""
    name = name.lower().strip()
    if name not in _PRUNING_REGISTRY:
        available = list(_PRUNING_REGISTRY.keys())
        raise KeyError(f"Unknown pruning method '{name}'. Available: {available}")
    return _PRUNING_REGISTRY[name]

def call_score_fn(method_name: str, kwargs: Dict[str, Any]):
    """
    Calls a registered score function safely, only passing the arguments
    that the specific function's signature explicitly accepts.
    """
    func = get_method(method_name)
    sig = inspect.signature(func)
    
    # If the function accepts **kwargs, pass all provided kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return func(**kwargs)
        
    # Otherwise, filter kwargs to match the function signature
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(**accepted)
