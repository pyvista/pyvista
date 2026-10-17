"""Benchmarks for wrapping and for converting between NumPy and VTK arrays."""

from __future__ import annotations

import numpy as np

import pyvista as pv
from pyvista.core.utilities.arrays import array_from_vtkmatrix
from pyvista.core.utilities.arrays import convert_array
from pyvista.core.utilities.arrays import convert_string_array
from pyvista.core.utilities.arrays import vtk_id_list_to_array
from pyvista.core.utilities.arrays import vtkmatrix_from_array
from pyvista.core.utilities.cells import numpy_to_idarr


def test_wrap_vtk_polydata(vtk_polydata, benchmark):
    """Wrap an unwrapped VTK polydata."""
    assert benchmark(pv.wrap, vtk_polydata).n_points


def test_pyvista_ndarray_from_vtk(sphere, benchmark):
    """Build a pyvista_ndarray from a VTK array."""
    vtk_array = sphere.GetPointData().GetArray('data')
    assert benchmark(pv.pyvista_ndarray, vtk_array).size


def test_convert_array_numpy_to_vtk(scalars, benchmark):
    """Convert a NumPy array to VTK."""
    assert benchmark(convert_array, scalars).GetNumberOfTuples()


def test_convert_string_array(string_array, benchmark):
    """Convert a NumPy string array to VTK."""
    assert benchmark(convert_string_array, string_array).GetNumberOfValues()


def test_vtk_id_list_to_array(vtk_id_list, benchmark):
    """Convert a VTK id list to NumPy."""
    assert benchmark(vtk_id_list_to_array, vtk_id_list).size


def test_array_from_vtkmatrix(benchmark):
    """Convert a VTK matrix to NumPy."""
    matrix = vtkmatrix_from_array(np.eye(4))
    assert benchmark(array_from_vtkmatrix, matrix).shape == (4, 4)


def test_vtkmatrix_from_array(benchmark):
    """Convert a NumPy matrix to VTK."""
    assert benchmark(vtkmatrix_from_array, np.eye(4)).GetElement(0, 0) == 1


def test_numpy_to_idarr(benchmark):
    """Convert a contiguous index array to a VTK id array."""
    indices = np.arange(10_000)
    assert benchmark(numpy_to_idarr, indices).GetNumberOfTuples()


def test_vtk_points(point_cloud, benchmark):
    """Build VTK points from a NumPy array."""
    assert benchmark(pv.vtk_points, point_cloud).GetNumberOfPoints()
