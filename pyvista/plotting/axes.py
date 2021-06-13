"""Module containing pyvista implementation of vtkAxes."""

import numpy as np

import pyvista
from pyvista import _vtk


class Axes(_vtk.vtkAxes):
    """PyVista wrapper for the VTK Axes class.

    Examples
    --------
    Create a axes at the pyvista module level

    >>> import pyvista
    >>> axes = pyvista.Axes()

    """

    def __init__(self, show_actor=False):
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

    @property
    def origin(self):
        """Origin of the axes in world coordinates.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.axes.origin
        (0.0, 0.0, 0.0)

        """
        return self.GetOrigin()

    @origin.setter
    def origin(self, value):
        """Set the origin of the camera.

        Examples
        --------
        >>> import pyvista
        >>> pl = pyvista.Plotter()
        >>> pl.axes.origin = (2.0, 1.0, 1.0)
        """
        self.SetOrigin(value)
