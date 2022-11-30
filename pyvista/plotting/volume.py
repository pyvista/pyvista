"""PyVista volume module."""
import pyvista as pv

from ._property import Property
from .mapper import _BaseMapper


class Volume(pv._vtk.vtkVolume):
    """Wrapper class for VTK volume."""

    def __init__(self):
        """Initialize volume."""
        super().__init__()

    @property
    def mapper(self) -> _BaseMapper:
        """Return or set the mapper of the volume.

        Examples
        --------
        Create an actor and assign a mapper to it.

        >>> import pyvista as pv
        >>> dataset = pv.Sphere()
        >>> actor = pv.Actor()
        >>> actor.mapper = pv.DataSetMapper(dataset)
        >>> actor.mapper  # doctest:+SKIP
        DataSetMapper (0x7f34dcec5040)
          Scalar visibility:           True
          Scalar range:                (0.0, 1.0)
          Interpolate before mapping:  False
          Scalar map mode:             default
          Color mode:                  direct
        <BLANKLINE>
        Attached dataset:
        PolyData (0x7f34dcec5f40)
          N Cells:  1680
          N Points: 842
          N Strips: 0
          X Bounds: -4.993e-01, 4.993e-01
          Y Bounds: -4.965e-01, 4.965e-01
          Z Bounds: -5.000e-01, 5.000e-01
          N Arrays: 1
        <BLANKLINE>

        """
        return self.GetMapper()

    @mapper.setter
    def mapper(self, obj):
        """Setter of the volume mapper."""
        return self.SetMapper(obj)

    @property
    def prop(self):
        """Return or set the property of this actor.

        Examples
        --------
        Modify the properties of an actor after adding a dataset to the plotter.

        >>> import pyvista as pv
        >>> pl = pv.Plotter()
        >>> actor = pl.add_mesh(pv.Sphere())
        >>> prop = actor.prop
        >>> prop.diffuse = 0.6
        >>> pl.show()

        """
        return self.GetProperty()

    @prop.setter
    def prop(self, obj: Property):
        """Setter of the volume properties."""
        self.SetProperty(obj)
