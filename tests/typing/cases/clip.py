"""Typing cases for :meth:`pyvista.DataObjectFilters.clip`.

What comes back is the input type for a `PolyData`, a `PointSet` or a `MultiBlock`,
and an `UnstructuredGrid` for every other dataset. `return_clipped` returns both
halves instead of one.
"""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv


def poly() -> pv.PolyData:
    """Return a sphere."""
    return pv.Sphere(theta_resolution=8, phi_resolution=8)


def image() -> pv.ImageData:
    """Return a small uniform grid."""
    return pv.ImageData(dimensions=(5, 5, 5))


def rectilinear() -> pv.RectilinearGrid:
    """Return a small rectilinear grid."""
    axis = np.arange(5, dtype=float)
    return pv.RectilinearGrid(axis, axis, axis)


def pointset() -> pv.PointSet:
    """Return a point cloud."""
    return pv.PointSet(poly().points)


def multiblock() -> pv.MultiBlock:
    """Return a composite of two meshes."""
    return pv.MultiBlock([poly(), image()])


assert_types(poly().clip(), pv.PolyData)
assert_types(poly().clip(normal='z'), pv.PolyData)
assert_types(poly().clip(crinkle=True), pv.PolyData)
assert_types(poly().clip(return_clipped=False), pv.PolyData)
assert_types(poly().clip(return_clipped=True), tuple[pv.PolyData, pv.PolyData])

assert_types(pointset().clip(), pv.PointSet)
assert_types(pointset().clip(return_clipped=True), tuple[pv.PointSet, pv.PointSet])

assert_types(multiblock().clip(), pv.MultiBlock)
assert_types(multiblock().clip(return_clipped=True), tuple[pv.MultiBlock, pv.MultiBlock])

assert_types(image().clip(), pv.UnstructuredGrid)
assert_types(rectilinear().clip(), pv.UnstructuredGrid)
assert_types(image().clip(crinkle=True), pv.UnstructuredGrid)
assert_types(image().clip(return_clipped=True), tuple[pv.UnstructuredGrid, pv.UnstructuredGrid])
