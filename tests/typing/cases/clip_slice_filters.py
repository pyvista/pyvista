"""Typing cases for the clip and slice filters.

Which class each filter gives back is checked at runtime in
``tests/core/test_dataobject_filters.py`` and ``tests/core/test_dataset_filters.py``;
these cases hold what a type checker infers from the same calls, one per overload.
"""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv
from pyvista import _vtk
from pyvista.core.utilities import generate_plane


def poly() -> pv.PolyData:
    """Return a sphere."""
    return pv.Sphere(theta_resolution=8, phi_resolution=8)


def pointset() -> pv.PointSet:
    """Return a point cloud."""
    return pv.PointSet(poly().points)


def image() -> pv.ImageData:
    """Return a small uniform grid."""
    return pv.ImageData(dimensions=(5, 5, 5), spacing=(0.25, 0.25, 0.25), origin=(-0.5, -0.5, -0.5))


def rectilinear() -> pv.RectilinearGrid:
    """Return a small rectilinear grid."""
    axis = np.linspace(-0.5, 0.5, 5)
    return pv.RectilinearGrid(axis, axis, axis)


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


def poly_with_scalars() -> pv.PolyData:
    """Return a sphere carrying point scalars."""
    mesh = poly()
    mesh.point_data['data'] = mesh.points[:, 2]
    return mesh


def image_with_scalars() -> pv.ImageData:
    """Return a uniform grid carrying point scalars."""
    mesh = image()
    mesh.point_data['data'] = np.linspace(0.0, 1.0, mesh.n_points)
    return mesh


def pointset_with_scalars() -> pv.PointSet:
    """Return a point cloud carrying point scalars."""
    mesh = pointset()
    mesh.point_data['data'] = mesh.points[:, 2]
    return mesh


def unstructured_with_scalars() -> pv.UnstructuredGrid:
    """Return an unstructured grid carrying point scalars."""
    mesh = unstructured()
    mesh.point_data['data'] = np.linspace(0.0, 1.0, mesh.n_points)
    return mesh


def explicit_structured() -> pv.ExplicitStructuredGrid:
    """Return a small explicit structured grid, which is not an UnstructuredGrid."""
    grid = structured()
    grid.dimensions = [5, 5, 5]
    return grid.cast_to_explicit_structured_grid()


def a_plane() -> _vtk.vtkPlane:
    """Return a plane through the origin."""
    return generate_plane((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def a_flag() -> bool:
    """Return a flag typed only as ``bool``, so the widened overloads apply."""
    return True


def a_line() -> pv.PolyData:
    """Return a polyline crossing the test meshes."""
    return pv.Line((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0), resolution=4)


def a_surface() -> pv.PolyData:
    """Return a closed surface enclosing the test meshes."""
    return pv.Sphere(radius=2.0, theta_resolution=8, phi_resolution=8)


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

# A box clip splits the cells it cuts, and clips a point cloud through its vertices
assert_types(poly().clip_box(), pv.PolyData)
assert_types(pointset().clip_box(), pv.PointSet)
assert_types(image().clip_box(), pv.UnstructuredGrid)
assert_types(multiblock().clip_box(), pv.MultiBlock)

# A slab clip follows the plane clip
assert_types(poly().clip_slab(thickness=0.2, normal='z'), pv.PolyData)
assert_types(pointset().clip_slab(thickness=0.2, normal='z'), pv.PointSet)
assert_types(image().clip_slab(thickness=0.2, normal='z'), pv.UnstructuredGrid)
assert_types(multiblock().clip_slab(thickness=0.2, normal='z'), pv.MultiBlock)

# Clipping by scalars follows the input class, and `both` hands back both halves
assert_types(poly_with_scalars().clip_scalar(value=0.0), pv.PolyData)
assert_types(poly_with_scalars().clip_scalar(value=0.0, both=True), tuple[pv.PolyData, pv.PolyData])
assert_types(poly_with_scalars().clip_scalar(value=0.0, inplace=True), pv.PolyData)
assert_types(pointset_with_scalars().clip_scalar(value=0.0), pv.PointSet)
assert_types(pointset_with_scalars().clip_scalar(value=0.0, both=True), tuple[pv.PointSet, pv.PointSet])
assert_types(unstructured_with_scalars().clip_scalar(value=0.0), pv.UnstructuredGrid)
assert_types(
    unstructured_with_scalars().clip_scalar(value=0.0, both=True),
    tuple[pv.UnstructuredGrid, pv.UnstructuredGrid],
)
assert_types(image_with_scalars().clip_scalar(value=0.0), pv.UnstructuredGrid)
assert_types(image_with_scalars().clip_scalar(value=0.0, both=True), tuple[pv.UnstructuredGrid, pv.UnstructuredGrid])
assert_types(poly_with_scalars().clip_scalar(value=0.0, both=a_flag()), pv.PolyData | tuple[pv.PolyData, pv.PolyData])

# Clipping by a surface follows the input class too
assert_types(poly().clip_surface(a_surface()), pv.PolyData)
assert_types(pointset().clip_surface(a_surface()), pv.PointSet)
assert_types(image().clip_surface(a_surface()), pv.UnstructuredGrid)
assert_types(poly().clip_closed_surface(), pv.PolyData)

# Slicing reduces any dataset to a surface, and a composite stays a composite
assert_types(poly().slice(), pv.PolyData)
assert_types(image().slice(), pv.PolyData)
assert_types(multiblock().slice(), pv.MultiBlock)
assert_types(image().slice_implicit(a_plane()), pv.PolyData)
assert_types(multiblock().slice_implicit(a_plane()), pv.MultiBlock)
assert_types(image().slice_along_line(a_line()), pv.PolyData)
assert_types(multiblock().slice_along_line(a_line()), pv.MultiBlock)

# These two collect their slices into a composite, whatever went in
assert_types(image().slice_orthogonal(), pv.MultiBlock)
assert_types(multiblock().slice_orthogonal(), pv.MultiBlock)
assert_types(image().slice_along_axis(n=2), pv.MultiBlock)
assert_types(multiblock().slice_along_axis(n=2), pv.MultiBlock)
