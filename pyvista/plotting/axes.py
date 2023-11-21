"""Module containing pyvista implementation of vtkAxes."""
from . import _vtk
from .actor import Actor
from .axes_actor import AxesActor


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

    symmetric : bool, optional
        If true, the axis continue to negative values.

    Examples
    --------
    Create an instance of axes at the pyvista module level.

    >>> import pyvista as pv
    >>> axes = pv.Axes()

    """

    def __init__(
        self, show_actor=False, actor_scale=1, line_width=1.0, symmetric=False
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
    def origin(self, value):  # numpydoc ignore=GL08
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
        self.axes_actor = None
        self.actor = None
        self.mapper = None
