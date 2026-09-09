"""Typing cases for the filters on :class:`pyvista.DataObjectFilters`.

Each case names what the filter returns for that input class, taken from running it.
"""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv


def poly() -> pv.PolyData:
    """Return a sphere."""
    return pv.Sphere(theta_resolution=8, phi_resolution=8)


def image() -> pv.ImageData:
    """Return a small uniform grid carrying point and cell data."""
    grid = pv.ImageData(dimensions=(5, 5, 5))
    grid.point_data['data'] = np.linspace(0.0, 1.0, grid.n_points)
    grid.cell_data['cells'] = np.linspace(0.0, 1.0, grid.n_cells)
    return grid


def unstructured() -> pv.UnstructuredGrid:
    """Return a tetrahedralized sphere."""
    return poly().delaunay_3d()


def pointset() -> pv.PointSet:
    """Return a point cloud."""
    return pv.PointSet(poly().points)


def multiblock() -> pv.MultiBlock:
    """Return a composite of two meshes."""
    return pv.MultiBlock([poly(), image()])


assert_types(image().extract_all_edges(), pv.PolyData)
assert_types(multiblock().extract_all_edges(), pv.MultiBlock)

# Cell centres are points, whatever went in
assert_types(image().cell_centers(), pv.PolyData)
assert_types(pointset().cell_centers(), pv.PolyData)
assert_types(multiblock().cell_centers(), pv.MultiBlock)

# `PolyData` overrides `triangulate`, so only the grids reach this one
assert_types(image().triangulate(), pv.UnstructuredGrid)
assert_types(multiblock().triangulate(), pv.MultiBlock)

# These hand back the class they were given
assert_types(poly().elevation(), pv.PolyData)
assert_types(image().elevation(), pv.ImageData)
assert_types(pointset().elevation(), pv.PointSet)
assert_types(multiblock().elevation(), pv.MultiBlock)
assert_types(image().sample(image()), pv.ImageData)
assert_types(poly().sample(image()), pv.PolyData)
assert_types(image().compute_cell_sizes(), pv.ImageData)
assert_types(unstructured().compute_cell_sizes(), pv.UnstructuredGrid)
assert_types(image().cell_data_to_point_data(), pv.ImageData)
assert_types(image().ctp(), pv.ImageData)
assert_types(image().point_data_to_cell_data(), pv.ImageData)
assert_types(image().ptc(), pv.ImageData)
assert_types(image().cell_validator(), pv.ImageData)
assert_types(poly().cell_validator(), pv.PolyData)
