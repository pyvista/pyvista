"""Pyvista specific errors."""


class MissingDataError(ValueError):
    """Exception when data is missing, e.g. no active scalars can be set."""


class AmbiguousDataError(ValueError):
    """Exception when data is ambiguous, e.g. multiple active scalars can be set."""
