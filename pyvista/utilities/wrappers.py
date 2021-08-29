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
>>> pyvista._wrappers['vtkPolyData'] = Foo
>>> uniform_grid = pyvista.UniformGrid()
>>> surface = uniform_grid.extract_surface()
>>> assert isinstance(surface, Foo)

"""

import pyvista

_wrappers = {
    'vtkExplicitStructuredGrid': pyvista.ExplicitStructuredGrid,
    'vtkUnstructuredGrid': pyvista.UnstructuredGrid,
    'vtkRectilinearGrid': pyvista.RectilinearGrid,
    'vtkStructuredGrid': pyvista.StructuredGrid,
    'vtkPolyData': pyvista.PolyData,
    'vtkImageData': pyvista.UniformGrid,
    'vtkStructuredPoints': pyvista.UniformGrid,
    'vtkMultiBlockDataSet': pyvista.MultiBlock,
    'vtkTable': pyvista.Table,
    # 'vtkParametricSpline': pyvista.Spline,
}
