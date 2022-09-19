"""PyVista specific errors."""


class MissingDataError(ValueError):
    """Exception when data is missing, e.g. no active scalars can be set."""

    def __init__(self, message='No data available.'):
        """Call the base class constructor with the custom message."""
        super().__init__(message)


class AmbiguousDataError(ValueError):
    """Exception when data is ambiguous, e.g. multiple active scalars can be set."""

    def __init__(self, message="Multiple data available."):
        """Call the base class constructor with the custom message."""
        super().__init__(message)
