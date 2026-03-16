import time
from functools import wraps
from typing import Any
from .exceptions import UnsupportedFrameworkError

def timer(func):
    """Measures and prints the execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        print(f"⏱️ {func.__name__} took {time.time()-start:.2f}s")
        return res
    return wrapper

def logger(msg):
    """Prints a professional logging header before executing the function."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f"\n--- {msg} ---")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def get_framework_adapter(model: Any, config: dict) -> Any:
    """
    Dynamically infers the framework based on the model type and returns
    the appropriate initialized adapter without strict hard-dependencies.
    """
    model_type_str = str(type(model)).lower()
    
    if "torch" in model_type_str:
        try:
            from ..backends.torch_backend import PyTorchAdapter
            return PyTorchAdapter(config)
        except ImportError as e:
            raise UnsupportedFrameworkError(f"Detected PyTorch model, but PyTorch backend failed to load: {e}")
            
    elif "keras" in model_type_str or "tensorflow" in model_type_str:
        try:
            from ..backends.keras_backend import KerasAdapter
            return KerasAdapter(config)
        except ImportError as e:
            raise UnsupportedFrameworkError(f"Detected Keras model, but Keras backend failed to load: {e}")
            
    raise UnsupportedFrameworkError(f"Could not infer framework for model of type {type(model)}")

def framework_dispatch(func):
    """
    A creative decorator that injects the correct `adapter` into the decorated
    function/method based on the `model` argument provided to it.
    
    The decorated function must accept an `adapter` keyword argument.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Find the model in args or kwargs
        model = kwargs.get('model')
        if model is None:
            # Usually the first arg after 'self' in class methods, or first arg in functions
            # Let's do a simple heuristic: find the first arg that isn't self/config
            for arg in args:
                type_str = str(type(arg)).lower()
                if "torch" in type_str or "keras" in type_str or "tensorflow" in type_str:
                    model = arg
                    break
                    
        if model is None:
            raise ValueError("@framework_dispatch requires a 'model' argument to infer the backend.")
            
        # Get config if available, else empty dict
        config = kwargs.get('config', getattr(args[0], 'config', {}) if args else {})
        
        adapter = get_framework_adapter(model, config)
        kwargs['adapter'] = adapter
        return func(*args, **kwargs)
        
    return wrapper
