"""Typing cases for :meth:`pyvista.DataObjectFilters.cell_centers`."""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv


def poly() -> pv.PolyData:
    """Return a sphere."""
    return pv.Sphere(theta_resolution=8, phi_resolution=8)


def image() -> pv.ImageData:
    """Return a small uniform grid."""
    return pv.ImageData(dimensions=(4, 4, 4))


def rectilinear() -> pv.RectilinearGrid:
    """Return a small rectilinear grid."""
    axis = np.arange(4, dtype=float)
    return pv.RectilinearGrid(axis, axis, axis)


def structured() -> pv.StructuredGrid:
    """Return a small structured grid."""
    axis = np.arange(4, dtype=float)
    x, y, z = np.meshgrid(axis, axis, axis, indexing='ij')
    return pv.StructuredGrid(x, y, z)


def unstructured() -> pv.UnstructuredGrid:
    """Return a small unstructured grid."""
    return image().cast_to_unstructured_grid()


def explicit_structured() -> pv.ExplicitStructuredGrid:
    """Return a small explicit structured grid."""
    grid = structured()
    grid.dimensions = [4, 4, 4]
    return grid.cast_to_explicit_structured_grid()


def pointset() -> pv.PointSet:
    """Return a point cloud."""
    return pv.PointSet(poly().points)


def multiblock() -> pv.MultiBlock:
    """Return a composite of two meshes."""
    return pv.MultiBlock([poly(), image()])


# Cell centres come back as points, whatever went in
assert_types(poly().cell_centers(), pv.PolyData)
assert_types(image().cell_centers(), pv.PolyData)
assert_types(rectilinear().cell_centers(), pv.PolyData)
assert_types(structured().cell_centers(), pv.PolyData)
assert_types(unstructured().cell_centers(), pv.PolyData)
assert_types(explicit_structured().cell_centers(), pv.PolyData)
assert_types(pointset().cell_centers(), pv.PolyData)
assert_types(multiblock().cell_centers(), pv.MultiBlock)
