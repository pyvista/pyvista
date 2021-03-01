"""Pyvista specific errors."""

CAMERA_ERROR_MESSAGE = """Invalid camera description
Camera description must be one of the following:

Iterable containing position, focal_point, and view up.  For example:
[(2.0, 5.0, 13.0), (0.0, 0.0, 0.0), (-0.7, -0.5, 0.3)]

Iterable containing a view vector.  For example:
[-1.0, 2.0, -5.0]

A string containing the plane orthogonal to the view direction.  For example:
'xy'
"""


class NotAllTrianglesError(ValueError):
    """Exception when a mesh does not contain all triangles."""

    def __init__(self, message='Mesh must consist of only triangles'):
        """Empty init."""
        ValueError.__init__(self, message)


class InvalidCameraError(ValueError):
    """Exception when passed an invalid camera."""

    def __init__(self, message=CAMERA_ERROR_MESSAGE):
        """Empty init."""
        ValueError.__init__(self, message)


class DeprecationError(RuntimeError):
    """Used for depreciated methods and functions."""

    def __init__(self, message='This feature has been depreciated'):
        """Empty init."""
        RuntimeError.__init__(self, message)


class VTKVersionError(RuntimeError):
    """Requested feature is not supported by the installed VTK version."""

    def __init__(self, message='The requested feature is not supported by the installed VTK version.'):
        """Empty init."""
        RuntimeError.__init__(self, message)
