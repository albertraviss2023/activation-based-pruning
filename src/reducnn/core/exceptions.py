"""Custom exceptions for the reducnn package.

This module defines a hierarchy of exceptions to provide clear and specific 
error messages for common failure modes in the structural pruning pipeline.
"""

class PruningError(Exception):
    """Base class for all exceptions in the reducnn package.
    
    Provides a common catch-all for any library-specific errors.
    """
    pass

class UnsupportedFrameworkError(PruningError):
    """Raised when an unsupported framework or model type is encountered.
    
    Typically occurs during framework inference if a model is not recognized 
    as a PyTorch or Keras object, or if required backends are missing.
    """
    pass

class SurgeryError(PruningError):
    """Raised when physical structural pruning (tensor shrinking) fails.
    
    This can happen due to architectural incompatibilities, dimension 
    mismatches, or unsupported layer types during the graph-rebuilding phase.
    """
    pass

class MethodRegistrationError(PruningError):
    """Raised when a custom pruning method is invalid or incorrectly registered.
    
    Occurs if a registered scoring function is not callable or if there is a 
    collision in the pruning registry.
    """
    pass
