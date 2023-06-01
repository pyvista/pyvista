"""Wrapper mapping.

Setting ``pyvista._wrappers`` allows for developers to override the default class used
to coerce a ``vtkDataSet`` into a pyvista object.  This is useful when creating a
subclass of a :class:`pyvista.DataSet` class.

Examples
--------
A user-defined Foo class is defined that extends the functionality of
:class:`pyvista.PolyData`.  This class is set as the default wrapper for
``vtkPolyData`` objects.

>>> import pyvista
>>> class Foo(pyvista.PolyData):
...     pass  # Extend PolyData here
...
>>> pyvista._wrappers['vtkPolyData'] = Foo
>>> uniform_grid = pyvista.UniformGrid()
>>> surface = uniform_grid.extract_surface()
>>> assert isinstance(surface, Foo)

"""
from .composite import MultiBlock
from .grid import RectilinearGrid, UniformGrid
from .objects import Table
from .pointset import ExplicitStructuredGrid, PointSet, PolyData, StructuredGrid, UnstructuredGrid

_wrappers = {
    'vtkExplicitStructuredGrid': ExplicitStructuredGrid,
    'vtkUnstructuredGrid': UnstructuredGrid,
    'vtkRectilinearGrid': RectilinearGrid,
    'vtkStructuredGrid': StructuredGrid,
    'vtkPolyData': PolyData,
    'vtkImageData': UniformGrid,
    'vtkStructuredPoints': UniformGrid,
    'vtkMultiBlockDataSet': MultiBlock,
    'vtkTable': Table,
    'vtkPointSet': PointSet,
    # 'vtkParametricSpline': pyvista.Spline,
}
