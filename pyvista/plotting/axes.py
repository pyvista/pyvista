"""Module containing pyvista implementation of :vtk:`vtkAxes`."""

from __future__ import annotations

from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core._vtk_utilities import DisableVtkSnakeCase
from pyvista.core.utilities.misc import _NoNewAttrMixin

from . import _vtk
from .actor import Actor
from .axes_actor import AxesActor


class Axes(_NoNewAttrMixin, DisableVtkSnakeCase, _vtk.vtkAxes):
    """PyVista wrapper for the VTK Axes class.

    Parameters
    ----------
    show_actor : bool, optional
        Hide or show the actor of these axes.  Default ``False``.

    actor_scale : float, optional
        Scale the size of the axes actor.  Default ``1``.

    line_width : float, optional
        Width of the axes lines.  Default ``1``.

    symmetric : bool, optional
        If true, the axis continue to negative values.

    Examples
    --------
    Create an instance of axes at the pyvista module level.

    >>> import pyvista as pv
    >>> axes = pv.Axes()

    """

    @_deprecate_positional_args
    def __init__(  # noqa: PLR0917
        self,
        show_actor: bool = False,  # noqa: FBT001, FBT002
        actor_scale=1,
        line_width=1.0,
        symmetric: bool = False,  # noqa: FBT001, FBT002
    ):  # numpydoc ignore=PR01,RT01
        """Initialize a new axes descriptor."""
        super().__init__()
        self.SetSymmetric(symmetric)
        # Add the axes mapper
        self.mapper = _vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(self.GetOutputPort())
        # Add the axes actor
        self.actor = Actor(mapper=self.mapper)
        self.axes_actor = AxesActor()
        self.actor.visibility = show_actor
        self.actor.scale = actor_scale
        self.actor.prop.line_width = line_width

    @property
    def origin(self):  # numpydoc ignore=RT01
        """Return or set th origin of the axes in world coordinates.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes = pv.Axes()
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
        self.SetOrigin(value)

    def show_actor(self):
        """Show an actor of axes.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes = pv.Axes()
        >>> axes.show_actor()

        """
        self.actor.visibility = True

    def hide_actor(self):
        """Hide an actor of axes.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes = pv.Axes()
        >>> axes.hide_actor()

        """
        self.actor.visibility = False

    def show_symmetric(self):
        """Show symmetric of axes.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes = pv.Axes()
        >>> axes.show_symmetric()

        """
        self.SymmetricOn()

    def hide_symmetric(self):
        """Hide symmetric of axes.

        Examples
        --------
        >>> import pyvista as pv
        >>> axes = pv.Axes()
        >>> axes.hide_symmetric()

        """
        self.SymmetricOff()

    def __del__(self):
        """Clean the attributes of the class."""
        self.axes_actor = None  # type: ignore[assignment]
        self.actor = None  # type: ignore[assignment]
        self.mapper = None  # type: ignore[assignment]
