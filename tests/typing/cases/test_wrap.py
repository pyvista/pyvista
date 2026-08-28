"""Static and runtime typing cases for :func:`pyvista.wrap`."""

from __future__ import annotations

import numpy as np
from trimesh import Trimesh
from typing_extensions import assert_type

import pyvista as pv
from pyvista import _vtk
from tests.typing.type_assertions import assert_runtime_type


def test_wrap_vtk_polydata() -> None:
    """A :vtk:`vtkPolyData` wraps to `PolyData`."""
    result = pv.wrap(_vtk.vtkPolyData())
    assert_type(result, pv.PolyData)
    assert_runtime_type(result, pv.PolyData)


def test_wrap_polydata() -> None:
    """A `PolyData` wraps to `PolyData`."""
    result = pv.wrap(pv.PolyData())
    assert_type(result, pv.PolyData)
    assert_runtime_type(result, pv.PolyData)


def test_wrap_vtk_structured_grid() -> None:
    """A :vtk:`vtkStructuredGrid` wraps to `StructuredGrid`."""
    result = pv.wrap(_vtk.vtkStructuredGrid())
    assert_type(result, pv.StructuredGrid)
    assert_runtime_type(result, pv.StructuredGrid)


def test_wrap_structured_grid() -> None:
    """A `StructuredGrid` wraps to `StructuredGrid`."""
    result = pv.wrap(pv.StructuredGrid())
    assert_type(result, pv.StructuredGrid)
    assert_runtime_type(result, pv.StructuredGrid)


def test_wrap_vtk_explicit_structured_grid() -> None:
    """A :vtk:`vtkExplicitStructuredGrid` wraps to `ExplicitStructuredGrid`."""
    result = pv.wrap(_vtk.vtkExplicitStructuredGrid())
    assert_type(result, pv.ExplicitStructuredGrid)
    assert_runtime_type(result, pv.ExplicitStructuredGrid)


def test_wrap_explicit_structured_grid() -> None:
    """An `ExplicitStructuredGrid` wraps to `ExplicitStructuredGrid`."""
    result = pv.wrap(pv.ExplicitStructuredGrid())
    assert_type(result, pv.ExplicitStructuredGrid)
    assert_runtime_type(result, pv.ExplicitStructuredGrid)


def test_wrap_vtk_unstructured_grid() -> None:
    """A :vtk:`vtkUnstructuredGrid` wraps to `UnstructuredGrid`."""
    result = pv.wrap(_vtk.vtkUnstructuredGrid())
    assert_type(result, pv.UnstructuredGrid)
    assert_runtime_type(result, pv.UnstructuredGrid)


def test_wrap_unstructured_grid() -> None:
    """An `UnstructuredGrid` wraps to `UnstructuredGrid`."""
    result = pv.wrap(pv.UnstructuredGrid())
    assert_type(result, pv.UnstructuredGrid)
    assert_runtime_type(result, pv.UnstructuredGrid)


def test_wrap_vtk_point_set() -> None:
    """A :vtk:`vtkPointSet` wraps to `PointSet`."""
    result = pv.wrap(_vtk.vtkPointSet())
    assert_type(result, pv.PointSet)
    assert_runtime_type(result, pv.PointSet)


def test_wrap_point_set() -> None:
    """A `PointSet` wraps to `PointSet`."""
    result = pv.wrap(pv.PointSet())
    assert_type(result, pv.PointSet)
    assert_runtime_type(result, pv.PointSet)


def test_wrap_vtk_rectilinear_grid() -> None:
    """A :vtk:`vtkRectilinearGrid` wraps to `RectilinearGrid`."""
    result = pv.wrap(_vtk.vtkRectilinearGrid())
    assert_type(result, pv.RectilinearGrid)
    assert_runtime_type(result, pv.RectilinearGrid)


def test_wrap_rectilinear_grid() -> None:
    """A `RectilinearGrid` wraps to `RectilinearGrid`."""
    result = pv.wrap(pv.RectilinearGrid())
    assert_type(result, pv.RectilinearGrid)
    assert_runtime_type(result, pv.RectilinearGrid)


def test_wrap_vtk_structured_points() -> None:
    """A :vtk:`vtkStructuredPoints` wraps to `ImageData`."""
    result = pv.wrap(_vtk.vtkStructuredPoints())
    assert_type(result, pv.ImageData)
    assert_runtime_type(result, pv.ImageData)


def test_wrap_vtk_image_data() -> None:
    """A :vtk:`vtkImageData` wraps to `ImageData`."""
    result = pv.wrap(_vtk.vtkImageData())
    assert_type(result, pv.ImageData)
    assert_runtime_type(result, pv.ImageData)


def test_wrap_image_data() -> None:
    """An `ImageData` wraps to `ImageData`."""
    result = pv.wrap(pv.ImageData())
    assert_type(result, pv.ImageData)
    assert_runtime_type(result, pv.ImageData)


def test_wrap_vtk_multi_block_data_set() -> None:
    """A :vtk:`vtkMultiBlockDataSet` wraps to `MultiBlock`."""
    result = pv.wrap(_vtk.vtkMultiBlockDataSet())
    assert_type(result, pv.MultiBlock)
    assert_runtime_type(result, pv.MultiBlock)


def test_wrap_multi_block() -> None:
    """A `MultiBlock` wraps to `MultiBlock`."""
    result = pv.wrap(pv.MultiBlock())
    assert_type(result, pv.MultiBlock)
    assert_runtime_type(result, pv.MultiBlock)


def test_wrap_vtk_table() -> None:
    """A :vtk:`vtkTable` wraps to `Table`."""
    result = pv.wrap(_vtk.vtkTable())
    assert_type(result, pv.Table)
    assert_runtime_type(result, pv.Table)


def test_wrap_table() -> None:
    """A `Table` wraps to `Table`."""
    result = pv.wrap(pv.Table())
    assert_type(result, pv.Table)
    assert_runtime_type(result, pv.Table)


def test_wrap_vtk_partitioned_data_set() -> None:
    """A :vtk:`vtkPartitionedDataSet` wraps to `PartitionedDataSet`."""
    result = pv.wrap(_vtk.vtkPartitionedDataSet())
    assert_type(result, pv.PartitionedDataSet)
    assert_runtime_type(result, pv.PartitionedDataSet)


def test_wrap_partitioned_data_set() -> None:
    """A `PartitionedDataSet` wraps to `PartitionedDataSet`."""
    result = pv.wrap(pv.PartitionedDataSet())
    assert_type(result, pv.PartitionedDataSet)
    assert_runtime_type(result, pv.PartitionedDataSet)


def test_wrap_numpy_array() -> None:
    """A float array wraps to `PolyData` or `ImageData`, depending on its shape."""
    result = pv.wrap(np.zeros(shape=(100, 3)))
    assert_type(result, pv.PolyData | pv.ImageData)
    assert_runtime_type(result, pv.PolyData | pv.ImageData)


def test_wrap_vtk_float_array() -> None:
    """A :vtk:`vtkFloatArray` wraps to `pyvista_ndarray`."""
    result = pv.wrap(_vtk.vtkFloatArray())
    assert_type(result, pv.pyvista_ndarray)
    assert_runtime_type(result, pv.pyvista_ndarray)


def test_wrap_none() -> None:
    """`None` wraps to `None`."""
    result = pv.wrap(None)
    assert_type(result, None)
    assert_runtime_type(result, None)


def test_wrap_trimesh() -> None:
    """A `Trimesh` wraps to `PolyData`."""
    result = pv.wrap(Trimesh())
    assert_type(result, pv.PolyData)
    assert_runtime_type(result, pv.PolyData)
