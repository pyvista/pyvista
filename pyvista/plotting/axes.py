"""Module containing pyvista implementation of vtkAxes."""

import numpy as np

import pyvista
from pyvista import _vtk


class Axes(_vtk.vtkAxes):
    """PyVista wrapper for the VTK Axes class.

    Parameters
    ----------
    show_actor : bool, optional
        Hide or show the actor of these axes.  Default ``False``.

    actor_scale : float, optional
        Scale the size of the axes actor.  Default ``1``.

    line_width : float, optional
        Width of the axes lines.  Default ``1``.

    Examples
    --------
    Create an instance of axes at the pyvista module level.

    >>> import pyvista
    >>> axes = pyvista.Axes()

    """

    def __init__(self, show_actor=False, actor_scale=1, line_width=1.0):
        """Initialize a new axes descriptor."""
        super().__init__()
        self.mapper = None
        self.actor = None

        # Add the axes mapper
        self.mapper = _vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(self.GetOutputPort())
        # Add the axes actor
        self.actor = _vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.SetVisibility(show_actor)
        self.actor.SetScale(actor_scale)
        prop = self.actor.GetProperty()
        prop.SetLineWidth(line_width)

    @property
    def origin(self):
        """Origin of the axes in world coordinates.

        Examples
        --------
        >>> import pyvista
        >>> axes = pyvista.Axes()
        >>> axes.origin
        (0.0, 0.0, 0.0)

        Set the origin of the camera.

        >>> axes.origin = (2.0, 1.0, 1.0)
        >>> axes.origin
        (2.0, 1.0, 1.0)

        """
        return self.GetOrigin()

    @origin.setter
    def origin(self, value):
        """Set the origin of the camera."""
        self.SetOrigin(value)
