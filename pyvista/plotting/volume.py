"""PyVista volume module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyvista._deprecate_positional_args import _deprecate_positional_args

from . import _vtk
from .prop3d import Prop3D

if TYPE_CHECKING:
    from typing_extensions import Self

    from .mapper import _BaseMapper
    from .volume_property import VolumeProperty


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
    def mapper(self) -> _BaseMapper:  # numpydoc ignore=RT01
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
        BoundsTuple(x_min = 0.0,
                    x_max = 9.0,
                    y_min = 0.0,
                    y_max = 9.0,
                    z_min = 0.0,
                    z_max = 9.0)

        """
        return self.GetMapper()  # type: ignore[return-value]

    @mapper.setter
    def mapper(self, obj):
        self.SetMapper(obj)

    @property
    def prop(self):  # numpydoc ignore=RT01
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
    def prop(self, obj: VolumeProperty):
        self.SetProperty(obj)

    @_deprecate_positional_args
    def copy(self: Self, deep: bool = True) -> Self:  # noqa: FBT001, FBT002
        """Create a copy of this volume.

        Parameters
        ----------
        deep : bool, default: True
            Create a shallow or deep copy of the volume. A deep copy will have a
            new property and mapper, while a shallow copy will use the mapper
            and property of this volume.

        Returns
        -------
        Volume
            Deep or shallow copy of this volume.

        Examples
        --------
        Create a volume of by adding it to a :class:`~pyvista.Plotter`
        and then copy the volume.

        >>> import pyvista as pv
        >>> mesh = pv.Wavelet()
        >>> pl = pv.Plotter()
        >>> volume = pl.add_volume(mesh, diffuse=0.5)
        >>> new_volume = volume.copy()

        Change the copy's properties. A deep copy is made by default, so the original
        volume is not affected.

        >>> new_volume.prop.diffuse = 1.0
        >>> new_volume.prop.diffuse
        1.0

        >>> volume.prop.diffuse
        0.5

        """
        new_actor = type(self)()
        if deep:
            if self.mapper is not None:
                new_actor.mapper = self.mapper.copy()
            new_actor.prop = self.prop.copy()
        else:
            new_actor.ShallowCopy(self)
        return new_actor
