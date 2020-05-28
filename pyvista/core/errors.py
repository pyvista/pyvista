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
