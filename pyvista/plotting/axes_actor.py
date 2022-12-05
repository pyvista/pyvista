"""Axes actor module."""
import pyvista as pv

from ._property import Property
from .mapper import _BaseMapper


class AxesActor(pv._vtk.vtkAxesActor):
    """Axes actor wrapper for VTK."""

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
        """Mapper getter."""
        return self.GetMapper()

    @mapper.setter
    def mapper(self, obj):
        """Mapper setter."""
        return self.SetMapper(obj)

    @property
    def prop(self):
        """Prop getter."""
        return self.GetProperty()

    @prop.setter
    def prop(self, obj: Property):
        """Prop setter."""
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
