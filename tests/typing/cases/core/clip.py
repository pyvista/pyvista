"""Typing cases for :meth:`pyvista.DataObjectFilters.clip`."""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv


def poly() -> pv.PolyData:
    """Return a sphere."""
    return pv.Sphere(theta_resolution=8, phi_resolution=8)


def pointset() -> pv.PointSet:
    """Return a point cloud."""
    return pv.PointSet(poly().points)


def image() -> pv.ImageData:
    """Return a small uniform grid."""
    return pv.ImageData(dimensions=(5, 5, 5), spacing=(0.25, 0.25, 0.25), origin=(-0.5, -0.5, -0.5))


def structured() -> pv.StructuredGrid:
    """Return a small structured grid."""
    axis = np.linspace(-0.5, 0.5, 5)
    x, y, z = np.meshgrid(axis, axis, axis, indexing='ij')
    return pv.StructuredGrid(x, y, z)


def unstructured() -> pv.UnstructuredGrid:
    """Return a small unstructured grid."""
    return image().cast_to_unstructured_grid()


def multiblock() -> pv.MultiBlock:
    """Return a composite of two meshes."""
    return pv.MultiBlock([poly(), image()])


def explicit_structured() -> pv.ExplicitStructuredGrid:
    """Return a small explicit structured grid, which is not an UnstructuredGrid."""
    grid = structured()
    grid.dimensions = [5, 5, 5]
    return grid.cast_to_explicit_structured_grid()


def a_flag() -> bool:
    """Return a flag typed only as ``bool``, so the widened overloads apply."""
    return True


# A plane clip keeps a surface a surface and a point cloud a point cloud
assert_types(poly().clip(), pv.PolyData)
assert_types(pointset().clip(), pv.PointSet)
assert_types(unstructured().clip(), pv.UnstructuredGrid)
assert_types(image().clip(), pv.UnstructuredGrid)
assert_types(multiblock().clip(), pv.MultiBlock)
# An ExplicitStructuredGrid is not an UnstructuredGrid, so it takes the DataSet overload
assert_types(explicit_structured().clip(), pv.UnstructuredGrid)

# `return_clipped` hands back both halves, of the same class
assert_types(poly().clip(return_clipped=True), tuple[pv.PolyData, pv.PolyData])
assert_types(pointset().clip(return_clipped=True), tuple[pv.PointSet, pv.PointSet])
assert_types(unstructured().clip(return_clipped=True), tuple[pv.UnstructuredGrid, pv.UnstructuredGrid])
assert_types(image().clip(return_clipped=True), tuple[pv.UnstructuredGrid, pv.UnstructuredGrid])
assert_types(multiblock().clip(return_clipped=True), tuple[pv.MultiBlock, pv.MultiBlock])

# A flag the caller computed gives both halves as one union
assert_types(poly().clip(return_clipped=a_flag()), pv.PolyData | tuple[pv.PolyData, pv.PolyData])
assert_types(pointset().clip(return_clipped=a_flag()), pv.PointSet | tuple[pv.PointSet, pv.PointSet])
assert_types(unstructured().clip(return_clipped=a_flag()), pv.UnstructuredGrid | tuple[pv.UnstructuredGrid, pv.UnstructuredGrid])
assert_types(image().clip(return_clipped=a_flag()), pv.UnstructuredGrid | tuple[pv.UnstructuredGrid, pv.UnstructuredGrid])
assert_types(multiblock().clip(return_clipped=a_flag()), pv.MultiBlock | tuple[pv.MultiBlock, pv.MultiBlock])

# Only the classes a clip can be copied back into accept `inplace`
assert_types(poly().clip(inplace=True), pv.PolyData)
assert_types(pointset().clip(inplace=True), pv.PointSet)
assert_types(unstructured().clip(inplace=True), pv.UnstructuredGrid)
