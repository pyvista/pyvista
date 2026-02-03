"""Wrapper mapping.

Setting ``pyvista._wrappers`` allows for developers to override the default class used
to coerce a :vtk:`vtkDataSet` into a pyvista object. This is useful when creating a
subclass of a :class:`pyvista.DataSet` class.

Examples
--------
A user-defined Foo class is defined that extends the functionality of
:class:`pyvista.PolyData`.  This class is set as the default wrapper for
:vtk:`vtkPolyData` objects.

>>> import pyvista as pv
>>> default_wrappers = pv._wrappers.copy()
>>> class Foo(pv.PolyData):
...     pass  # Extend PolyData here
>>> pv._wrappers['vtkPolyData'] = Foo
>>> image = pv.ImageData()
>>> surface = image.extract_surface(algorithm='auto')
>>> assert isinstance(surface, Foo)
>>> pv._wrappers = default_wrappers  # reset back to default

"""

from __future__ import annotations

from typing import TypeVar

from . import _vtk_core as _vtk
from .composite import MultiBlock
from .grid import ImageData
from .grid import RectilinearGrid
from .objects import Table
from .partitioned import PartitionedDataSet
from .pointset import ExplicitStructuredGrid
from .pointset import PointSet
from .pointset import PolyData
from .pointset import StructuredGrid
from .pointset import UnstructuredGrid

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
    'vtkPartitionedDataSet': PartitionedDataSet,
    # 'vtkParametricSpline': pyvista.Spline,
}

_WrappableVTKDataObjectType = TypeVar(  # noqa: PYI018
    '_WrappableVTKDataObjectType',
    _vtk.vtkExplicitStructuredGrid,
    _vtk.vtkUnstructuredGrid,
    _vtk.vtkRectilinearGrid,
    _vtk.vtkStructuredGrid,
    _vtk.vtkPolyData,
    _vtk.vtkImageData,
    _vtk.vtkStructuredPoints,
    _vtk.vtkMultiBlockDataSet,
    _vtk.vtkTable,
    _vtk.vtkPoints,
    _vtk.vtkPartitionedDataSet,
)
