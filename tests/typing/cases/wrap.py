"""Typing cases for :func:`pyvista.wrap`."""

from __future__ import annotations

import numpy as np
from trimesh import Trimesh

import pyvista as pv
from pyvista import _vtk
from tests.typing.typeassert import assert_types

assert_types(pv.wrap(_vtk.vtkPolyData()), pv.PolyData)
assert_types(pv.wrap(pv.PolyData()), pv.PolyData)

assert_types(pv.wrap(_vtk.vtkStructuredGrid()), pv.StructuredGrid)
assert_types(pv.wrap(pv.StructuredGrid()), pv.StructuredGrid)

assert_types(pv.wrap(_vtk.vtkExplicitStructuredGrid()), pv.ExplicitStructuredGrid)
assert_types(pv.wrap(pv.ExplicitStructuredGrid()), pv.ExplicitStructuredGrid)

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
