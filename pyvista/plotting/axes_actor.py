"""Axes actor module."""
import pyvista as pv

from ._property import Property
from .mapper import _BaseMapper


class AxesActor(pv._vtk.vtkAxesActor):
    """Axes actor wrapper for vtkAxesActor.

    Hybrid 2D/3D actor used to represent 3D axes in a scene. The user
    can define the geometry to use for the shaft or the tip, and the
    user can set the text for the three axes. To see full customization
    options, refer to `VTK axes actor documentation
    <https://vtk.org/doc/nightly/html/classvtkAxesActor.html#details>`

    Examples
    --------
    Customize the axis shaft color and shape.

    >>> import pyvista as pv
    >>> axes = pv.Axes()
    >>> axes.axes_actor.GetZAxisShaftProperty().SetColor(0, 1, 1)
    >>> axes.axes_actor.SetShaftTypeToCylinder()
    >>> pl = pv.Plotter()
    >>> pl.add_actor(axes.axes_actor)
    >>> pl.add_mesh(pv.Sphere())
    >>> pl.show()

    """

    def __init__(self, mapper=None, prop=None):
        """Initialize actor."""
        super().__init__()

        def __init__(self, mapper=None, prop=None):
            """Initialize actor."""
            super().__init__()
            if mapper is not None:
                self.mapper = mapper
            if prop is None:
                self.prop = Property()

    @property
    def mapper(self) -> _BaseMapper:
        """Return or set actor set of properties.

        Examples
        --------
        Create an actor using the :class:`pyvista.Plotter` and then change the
        visibility of the actor.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()

        """
        return self.GetMapper()

    @mapper.setter
    def mapper(self, obj):
        return self.SetMapper(obj)

    @property
    def prop(self):
        """Return or set AxesActor set of properties.

        Examples
        --------
        Create an actor using the :class:`pyvista.Plotter` and then change the
        visibility of the actor.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(pv.Sphere())

        """
        return self.GetProperty()

    @prop.setter
    def prop(self, obj: Property):
        self.SetProperty(obj)

    @property
    def visibility(self) -> bool:
        """Return or set actor visibility.

        Examples
        --------
        Create an actor using the :class:`pyvista.Plotter` and then change the
        visibility of the actor.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(pv.Sphere())
        >>> actor.visibility = False
        >>> actor.visibility
        False

        """
        return bool(self.GetVisibility())

    @visibility.setter
    def visibility(self, value: bool):
        return self.SetVisibility(value)
