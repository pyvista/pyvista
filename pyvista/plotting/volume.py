"""PyVista volume module."""
import pyvista as pv

from ._property import Property
from .mapper import _BaseMapper


class Volume(pv._vtk.vtkVolume):
    """Wrapper class for VTK volume.

    This class represents a volume in a rendered scene.

    """

    def __init__(self):
        """Initialize volume."""
        super().__init__()

    @property
    def mapper(self) -> _BaseMapper:
        """Return or set the mapper of the volume.

        Examples
        --------
        Create an volume and get its mapper.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> vol = examples.download_knee_full()
        >>> p = pv.Plotter(notebook=0)
        >>> actor = p.add_volume(vol, cmap="bone", opacity="sigmoid")
        >>> print(actor.mapper.GetBounds())
        (0.0, 149.661, 0.0, 178.581, 0.0, 200.0)
        """
        return self.GetMapper()

    @mapper.setter
    def mapper(self, obj):
        """Setter of the volume mapper.

        Examples
        --------
        Create an volume and set its mapper.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> vol = examples.download_knee_full()
        >>> p = pv.Plotter(notebook=0)
        >>> actor = p.add_volume(vol, cmap="bone", opacity="sigmoid")
        >>> actor.mapper.lookup_table.cmap = 'viridis'

        """
        return self.SetMapper(obj)

    @property
    def prop(self):
        """Return or set the property of this actor.

        Examples
        --------
        Create an volume and get its properties.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> pl = pv.Plotter()
        >>> vol = examples.download_knee_full()
        >>> actor = pl.add_volume(vol, cmap="bone", opacity="sigmoid")
        >>> print(actor.prop.GetShade())
        0

        """
        return self.GetProperty()

    @prop.setter
    def prop(self, obj: Property):
        """Setter of the volume properties.

        Examples
        --------
        Create an volume and set its mapper.

        >>> import pyvista as pv
        >>> from pyvista import examples
        >>> vol = examples.download_knee_full()
        >>> p = pv.Plotter(notebook=0)
        >>> actor = p.add_volume(vol, cmap="bone", opacity="sigmoid")
        >>> actor.prop.SetShade(True)

        """
        self.SetProperty(obj)
