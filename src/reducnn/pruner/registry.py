import inspect
from typing import Callable, Dict, Any, Tuple, List, Optional
from ..core.exceptions import MethodRegistrationError

# Stores the mapping of (method_name, framework) to the actual implementation.
# The 'framework' key can be "torch", "keras", or "global" (for agnostic methods).
_PRUNING_REGISTRY: Dict[Tuple[str, str], Callable[..., Any]] = {}
# Stores metadata about the method, such as supported scopes ('local', 'global').
_METHOD_METADATA: Dict[Tuple[str, str], Dict[str, Any]] = {}

def list_methods(framework: Optional[str] = None, include_global: bool = True) -> List[Dict[str, Any]]:
    """Lists registered pruning methods with their metadata.

    Args:
        framework: Optional framework filter such as "torch" or "keras".
        include_global: When filtering by framework, include framework-agnostic
            methods registered as "global".

    Returns:
        A stable list of dictionaries with ``name``, ``framework``, and ``supported_scopes``.
    """
    fw = framework.lower().strip() if framework else None
    rows: List[Dict[str, Any]] = []
    for key in sorted(_PRUNING_REGISTRY.keys()):
        name, method_framework = key
        if fw is not None and method_framework != fw:
            if not (include_global and method_framework == "global"):
                continue
        meta = _METHOD_METADATA.get(key, {})
        rows.append({
            "name": name, 
            "framework": method_framework,
            "supported_scopes": meta.get("supported_scopes", ["local", "global"])
        })
    return rows

def list_method_names(framework: Optional[str] = None, include_global: bool = True) -> List[str]:
    """Lists unique registered method names for a framework."""
    return sorted({row["name"] for row in list_methods(framework, include_global=include_global)})

def register_method(name: str, framework: str = "global", supported_scopes: Optional[List[str]] = None) -> Callable:
    """Decorator to register a custom pruning score function with scope metadata.

    Args:
        name (str): The unique name of the pruning method.
        framework (str): The framework context ('torch', 'keras', or 'global').
        supported_scopes (Optional[List[str]]): List of supported pruning scopes 
            (e.g., ['local'], ['global'], or ['local', 'global']). 
            Defaults to both if not specified.

    Returns:
        Callable: The decorator function.
    """
    def decorator(func: Callable) -> Callable:
        if not callable(func):
            raise MethodRegistrationError(f"Pruning method {name} must be callable.")
        
        key = (name.lower().strip(), framework.lower().strip())
        _PRUNING_REGISTRY[key] = func
        _METHOD_METADATA[key] = {
            "supported_scopes": supported_scopes or ["local", "global"]
        }
        return func
    return decorator

def get_method(name: str, framework: str) -> Callable:
    """Retrieves a pruning method from the registry for a specific framework.

    The retrieval logic follows a fallback pattern:
    1. Look for a framework-specific implementation.
    2. Fall back to a 'global' (framework-agnostic) implementation.

    Args:
        name (str): The name of the pruning method.
        framework (str): The framework context ('torch' or 'keras').

    Returns:
        Callable: The registered pruning score function.

    Raises:
        KeyError: If the pruning method is not found in the registry for 
            either the specific framework or the global scope.
    """
    name = name.lower().strip()
    framework = framework.lower().strip()
    
    # 1. Attempt to find a framework-specific implementation
    key = (name, framework)
    if key in _PRUNING_REGISTRY:
        return _PRUNING_REGISTRY[key]
    
    # 2. Fall back to the global (framework-agnostic) implementation
    global_key = (name, "global")
    if global_key in _PRUNING_REGISTRY:
        return _PRUNING_REGISTRY[global_key]
        
    # If not found, list available methods for the requested framework/global to help debugging
    available = [f"{k[0]}" for k in _PRUNING_REGISTRY.keys() if k[1] == framework or k[1] == "global"]
    raise KeyError(f"Unknown pruning method '{name}' for framework '{framework}'. "
                   f"Available for this framework: {available}")

def call_score_fn(method_name: str, framework: str, kwargs: Dict[str, Any]) -> Any:
    """Safely calls a registered score function, handling argument matching.

    Introspects the target function's signature to ensure only supported 
    arguments from `kwargs` are passed, preventing `TypeError`.

    Args:
        method_name (str): The name of the pruning method to call.
        framework (str): The framework context.
        kwargs (Dict[str, Any]): All available arguments (e.g., 'layer', 'model', 'loader').

    Returns:
        Any: The result of the score function (typically a NumPy array of scores).
    """
    func = get_method(method_name, framework)
    # Introspect the function signature
    sig = inspect.signature(func)
    
    # If the function accepts arbitrary keyword arguments (**kwargs), pass everything
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return func(**kwargs)
        
    # Otherwise, filter the provided kwargs to match only what the function accepts
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(**accepted)
