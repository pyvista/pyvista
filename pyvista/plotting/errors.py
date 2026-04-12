"""Plotting errors."""

from __future__ import annotations

CAMERA_ERROR_MESSAGE = """Invalid camera description
Camera description must be one of the following:

Iterable containing position, focal_point, and view up.  For example:
[(2.0, 5.0, 13.0), (0.0, 0.0, 0.0), (-0.7, -0.5, 0.3)]

Iterable containing a view vector.  For example:
[-1.0, 2.0, -5.0]

A string containing the plane orthogonal to the view direction.  For example:
'xy'
"""


class InvalidCameraError(ValueError):  # numpydoc ignore=PR01
    """Exception when passed an invalid camera."""

    def __init__(self, message=CAMERA_ERROR_MESSAGE):
        """Call the base class constructor with the custom message."""
        super().__init__(message)


class RenderWindowUnavailable(RuntimeError):  # numpydoc ignore=PR01 # noqa: N818
    """Exception when the render window is not available."""

    def __init__(self, message='Render window is not available.'):
        """Call the base class constructor with the custom message."""
        super().__init__(message)


class PyVistaPickingError(RuntimeError):
    """General picking error class."""
