"""PyVista volume module."""
from . import _vtk
from ._property import Property
from .mapper import _BaseMapper
from .prop3d import Prop3D


class Volume(Prop3D, _vtk.vtkVolume):
    """Wrapper class for VTK volume.

    This class represents a volume in a rendered scene. It inherits
    functions related to the volume's position, orientation and origin
    from Prop3D.

    """

    def __init__(self):
        """Initialize volume."""
        super().__init__()

    @property
    def mapper(self) -> _BaseMapper:
        """Return or set the mapper of the volume.

        Examples
        --------
        Add a volume to a :class:`pyvista.Plotter` and get its mapper.

        >>> import pyvista as pv
        >>> vol = pv.ImageData(dimensions=(10, 10, 10))
        >>> vol['scalars'] = 255 - vol.z * 25
        >>> pl = pv.Plotter()
        >>> actor = pl.add_volume(vol)
        >>> actor.mapper.bounds
        (0.0, 9.0, 0.0, 9.0, 0.0, 9.0)
        """
        return self.GetMapper()

    @mapper.setter
    def mapper(self, obj):
        return self.SetMapper(obj)

    @property
    def prop(self):
        """Return or set the property of this actor.

        Examples
        --------
        Create an volume and get its properties.

        >>> import pyvista as pv
        >>> vol = pv.ImageData(dimensions=(10, 10, 10))
        >>> vol['scalars'] = 255 - vol.z * 25
        >>> pl = pv.Plotter()
        >>> actor = pl.add_volume(vol)
        >>> actor.prop.GetShade()
        0

        """
        return self.GetProperty()

    @prop.setter
    def prop(self, obj: Property):
        self.SetProperty(obj)
