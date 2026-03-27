import time
from functools import wraps
from typing import Any, Callable
from .exceptions import UnsupportedFrameworkError

def timer(func: Callable) -> Callable:
    """Measures and prints the execution time of a function.

    Args:
        func (Callable): The function to be timed.

    Returns:
        Callable: The wrapped function that includes timing logic.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        # Calculate and print the elapsed time in seconds
        print(f"⏱️ {func.__name__} took {time.time()-start:.2f}s")
        return res
    return wrapper

def logger(msg: str) -> Callable:
    """Prints a professional logging header before executing the function.

    Args:
        msg (str): The message to be displayed in the logging header.

    Returns:
        Callable: A decorator that wraps the target function with logging.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Display the log message as a section header
            print(f"\n--- {msg} ---")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def get_framework_adapter(model: Any, config: dict) -> Any:
    """Dynamically infers the framework based on the model type.

    Returns the appropriate initialized adapter without strict hard-dependencies.
    This enables the core logic to remain framework-agnostic.

    Args:
        model (Any): The model object (PyTorch, Keras, or TensorFlow).
        config (dict): Configuration dictionary for the adapter initialization.

    Returns:
        Any: An instance of the corresponding framework adapter (PyTorchAdapter or KerasAdapter).

    Raises:
        UnsupportedFrameworkError: If the model type cannot be mapped to a supported framework
            or if the required backend fails to load.
    """
    model_type_str = str(type(model)).lower()
    
    # Check if it's a PyTorch model
    if "torch" in model_type_str:
        try:
            from ..backends.torch_backend import PyTorchAdapter
            return PyTorchAdapter(config)
        except ImportError as e:
            raise UnsupportedFrameworkError(f"Detected PyTorch model, but PyTorch backend failed to load: {e}")
            
    # Check if it's a Keras or TensorFlow model
    elif "keras" in model_type_str or "tensorflow" in model_type_str:
        try:
            from ..backends.keras_backend import KerasAdapter
            return KerasAdapter(config)
        except ImportError as e:
            raise UnsupportedFrameworkError(f"Detected Keras model, but Keras backend failed to load: {e}")
            
    raise UnsupportedFrameworkError(f"Could not infer framework for model of type {type(model)}")

def framework_dispatch(func: Callable) -> Callable:
    """Decorator that injects the correct `adapter` into the decorated function.

    The backend is inferred based on the `model` argument provided to the function.
    The decorated function must accept an `adapter` keyword argument.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The wrapped function with automatic adapter injection.

    Raises:
        ValueError: If a 'model' argument cannot be found in the arguments or keyword arguments.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Allow callers to explicitly inject an adapter. This is important for
        # orchestration paths that already resolved backend selection.
        if kwargs.get("adapter") is not None:
            return func(*args, **kwargs)

        # Attempt to find the model in keyword arguments first
        model = kwargs.get('model')
        if model is None:
            # Heuristic: search through positional arguments for a framework-specific object
            # Usually the first arg after 'self' in class methods, or first arg in functions
            for arg in args:
                type_str = str(type(arg)).lower()
                if "torch" in type_str or "keras" in type_str or "tensorflow" in type_str:
                    model = arg
                    break
                    
        if model is None:
            raise ValueError("@framework_dispatch requires a 'model' argument to infer the backend.")
            
        # Extract configuration if available, defaulting to an empty dictionary
        # It checks kwargs first, then attempts to find it as an attribute on the first argument (often 'self')
        config = kwargs.get('config', getattr(args[0], 'config', {}) if args else {})
        
        # Initialize the appropriate adapter for the detected framework
        adapter = get_framework_adapter(model, config)
        kwargs['adapter'] = adapter
        
        return func(*args, **kwargs)
        
    return wrapper
