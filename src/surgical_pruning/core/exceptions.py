"""Custom exceptions for the surgical_pruning package."""

class PruningError(Exception):
    """Base class for exceptions in this package."""
    pass

class UnsupportedFrameworkError(PruningError):
    """Raised when an unsupported framework or model type is encountered."""
    pass

class SurgeryError(PruningError):
    """Raised when physical structural pruning fails."""
    pass

class MethodRegistrationError(PruningError):
    """Raised when a custom pruning method is invalid or incorrectly registered."""
    pass
