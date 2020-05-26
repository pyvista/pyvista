"""Pyvista specific errors."""


class NotAllTrianglesError(ValueError):
    """Exception when a mesh does not contain all triangles."""

    def __init__(self, message='Mesh must consist of only triangles'):
        """Empty init."""
        Exception.__init__(self, message)
