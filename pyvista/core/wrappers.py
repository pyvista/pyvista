"""Wrapper mapping.

Setting ``pyvista._wrappers`` allows for developers to override the default class used
to coerce a ``vtkDataSet`` into a pyvista object.  This is useful when creating a
subclass of a :class:`pyvista.DataSet` class.

Examples
--------
A user-defined Foo class is defined that extends the functionality of
:class:`pyvista.PolyData`.  This class is set as the default wrapper for
``vtkPolyData`` objects.

>>> import pyvista as pv
>>> default_wrappers = pv._wrappers.copy()
>>> class Foo(pv.PolyData):
...     pass  # Extend PolyData here
...
>>> pv._wrappers['vtkPolyData'] = Foo
>>> image = pv.ImageData()
>>> surface = image.extract_surface()
>>> assert isinstance(surface, Foo)
>>> pv._wrappers = default_wrappers  # reset back to default

"""
from .composite import MultiBlock
from .grid import ImageData, RectilinearGrid
from .objects import Table
from .pointset import ExplicitStructuredGrid, PointSet, PolyData, StructuredGrid, UnstructuredGrid

_wrappers = {
    'vtkExplicitStructuredGrid': ExplicitStructuredGrid,
    'vtkUnstructuredGrid': UnstructuredGrid,
    'vtkRectilinearGrid': RectilinearGrid,
    'vtkStructuredGrid': StructuredGrid,
    'vtkPolyData': PolyData,
    'vtkImageData': ImageData,
    'vtkStructuredPoints': ImageData,
    'vtkMultiBlockDataSet': MultiBlock,
    'vtkTable': Table,
    'vtkPointSet': PointSet,
    # 'vtkParametricSpline': pyvista.Spline,
}
