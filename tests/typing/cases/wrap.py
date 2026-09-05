"""Typing cases for :func:`pyvista.wrap`."""

from __future__ import annotations

import meshio
import numpy as np
from trimesh import Trimesh
from type_assert import assert_types

import pyvista as pv
from pyvista import _vtk
from pyvista import examples


def as_data_set() -> _vtk.vtkDataSet:
    """Return a dataset typed only as the VTK base class."""
    return _vtk.vtkPolyData()


def as_data_object() -> _vtk.vtkDataObject:
    """Return a data object typed only as the VTK base class."""
    return _vtk.vtkTable()


def as_meshio_mesh() -> meshio.Mesh:
    """Return a single-triangle meshio mesh."""
    return meshio.Mesh(points=np.zeros((3, 3)), cells=[('triangle', np.array([[0, 1, 2]]))])


SKIP_RUNTIME = {
    'pv.wrap(_vtk.vtkExplicitStructuredGrid())': 'VTK segfaults on an empty grid',
}

assert_types(pv.wrap(_vtk.vtkPolyData()), pv.PolyData)
assert_types(pv.wrap(pv.PolyData()), pv.PolyData)

assert_types(pv.wrap(_vtk.vtkStructuredGrid()), pv.StructuredGrid)
assert_types(pv.wrap(pv.StructuredGrid()), pv.StructuredGrid)

assert_types(pv.wrap(_vtk.vtkExplicitStructuredGrid()), pv.ExplicitStructuredGrid)
assert_types(pv.wrap(examples.load_explicit_structured()), pv.ExplicitStructuredGrid)

assert_types(pv.wrap(_vtk.vtkUnstructuredGrid()), pv.UnstructuredGrid)
assert_types(pv.wrap(pv.UnstructuredGrid()), pv.UnstructuredGrid)

assert_types(pv.wrap(_vtk.vtkPointSet()), pv.PointSet)
assert_types(pv.wrap(pv.PointSet()), pv.PointSet)

assert_types(pv.wrap(_vtk.vtkRectilinearGrid()), pv.RectilinearGrid)
assert_types(pv.wrap(pv.RectilinearGrid()), pv.RectilinearGrid)

assert_types(pv.wrap(_vtk.vtkStructuredPoints()), pv.ImageData)
assert_types(pv.wrap(_vtk.vtkImageData()), pv.ImageData)
assert_types(pv.wrap(pv.ImageData()), pv.ImageData)

assert_types(pv.wrap(_vtk.vtkMultiBlockDataSet()), pv.MultiBlock)
assert_types(pv.wrap(pv.MultiBlock()), pv.MultiBlock)

assert_types(pv.wrap(_vtk.vtkTable()), pv.Table)
assert_types(pv.wrap(pv.Table()), pv.Table)

assert_types(pv.wrap(_vtk.vtkPartitionedDataSet()), pv.PartitionedDataSet)
assert_types(pv.wrap(pv.PartitionedDataSet()), pv.PartitionedDataSet)

assert_types(pv.wrap(np.zeros(shape=(100, 3))), pv.PolyData | pv.ImageData)
assert_types(pv.wrap(_vtk.vtkFloatArray()), pv.pyvista_ndarray)
assert_types(pv.wrap(None), None)
assert_types(pv.wrap(Trimesh()), pv.PolyData)
assert_types(pv.wrap(as_meshio_mesh()), pv.UnstructuredGrid)

# The catch-alls, reached only by an argument widened to a VTK base class
assert_types(pv.wrap(as_data_set()), pv.DataSet)
assert_types(pv.wrap(as_data_object()), pv.DataObject)

# `validate` does not change what comes back
assert_types(pv.wrap(pv.PolyData(), validate=True), pv.PolyData)
assert_types(pv.wrap(_vtk.vtkTable(), validate=False), pv.Table)
