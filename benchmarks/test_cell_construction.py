"""Benchmarks for building cell arrays and datasets from NumPy input."""

from __future__ import annotations

import numpy as np

import pyvista as pv
from pyvista.core.utilities.cells import create_mixed_cells


def test_cell_array_from_regular_cells(triangles, benchmark):
    """Build a cell array from equal-width cells."""
    assert benchmark(pv.CellArray.from_regular_cells, triangles).GetNumberOfCells()


def test_polydata_from_regular_faces(triangle_points, triangles, benchmark):
    """Build a polydata from points and equal-width faces."""
    assert benchmark(pv.PolyData.from_regular_faces, triangle_points, triangles).n_cells


def test_create_mixed_cells(benchmark):
    """Build a mixed cell array from ragged connectivity."""
    ragged = {pv.CellType.POLYGON: [np.arange(i % 5 + 3) for i in range(2000)]}
    assert benchmark(create_mixed_cells, ragged)[1].size


def test_polydata_from_points(point_cloud, benchmark):
    """Build a point cloud polydata."""
    assert benchmark(pv.PolyData, point_cloud).n_points
