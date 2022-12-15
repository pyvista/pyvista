"""Axes actor module."""
import pyvista as pv


class AxesActor(pv._vtk.vtkAxesActor):
    """Axes actor wrapper for vtkAxesActor.

    Hybrid 2D/3D actor used to represent 3D axes in a scene. The user
    can define the geometry to use for the shaft or the tip, and the
    user can set the text for the three axes. To see full customization
    options, refer to `vtkAxesActor Details
    <https://vtk.org/doc/nightly/html/classvtkAxesActor.html#details>`

    Examples
    --------
    Customize the axis shaft color and shape.

    >>> import pyvista as pv
    >>> axes = pv.Axes()
    >>> axes.axes_actor.GetZAxisShaftProperty().SetColor(0, 1, 1)
    >>> axes.axes_actor.SetShaftTypeToCylinder()
    >>> pl = pv.Plotter()
    >>> _ = pl.add_actor(axes.axes_actor)
    >>> _ = pl.add_mesh(pv.Sphere())
    >>> pl.show()

    """

    def __init__(self, mapper=None, prop=None):
        """Initialize actor."""
        super().__init__()

    @property
    def visibility(self) -> bool:
        """Return or set AxesActor visibility.

        Examples
        --------
        Create an Axes object and then change the
        visibility of its AxesActor.

        >>> import pyvista as pv
        >>> axes = pv.Axes()
        >>> axes.axes_actor.visibility
        True

        """
        return bool(self.GetVisibility())

    @visibility.setter
    def visibility(self, value: bool):
        return self.SetVisibility(value)
