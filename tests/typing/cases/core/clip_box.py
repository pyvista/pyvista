"""Typing cases for :meth:`pyvista.DataObjectFilters.clip_box`."""

from __future__ import annotations

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


def unstructured() -> pv.UnstructuredGrid:
    """Return a small unstructured grid."""
    return image().cast_to_unstructured_grid()


def multiblock_pointset() -> pv.MultiBlock[pv.PointSet]:
    """Return a composite declared to hold only `PointSet`."""
    return pv.MultiBlock([pointset()])


def multiblock_unstructured() -> pv.MultiBlock[pv.UnstructuredGrid]:
    """Return a composite declared to hold only `UnstructuredGrid`."""
    return pv.MultiBlock([unstructured()])


def multiblock_poly() -> pv.MultiBlock[pv.PolyData]:
    """Return a composite declared to hold only `PolyData`."""
    return pv.MultiBlock([poly()])


def multiblock_image() -> pv.MultiBlock[pv.ImageData]:
    """Return a composite declared to hold only `ImageData`."""
    return pv.MultiBlock([image()])


def multiblock() -> pv.MultiBlock:
    """Return a composite of two meshes."""
    return pv.MultiBlock([poly(), image()])


def multiblock_optional_poly() -> pv.MultiBlock[pv.PolyData | None]:
    """Return a composite whose blocks may be missing."""
    return pv.MultiBlock([poly(), None])


def multiblock_optional_image() -> pv.MultiBlock[pv.ImageData | None]:
    """Return a composite of grids whose blocks may be missing."""
    return pv.MultiBlock([image(), None])


def multiblock_optional_pointset() -> pv.MultiBlock[pv.PointSet | None]:
    """Return a composite of point clouds whose blocks may be missing."""
    return pv.MultiBlock([pointset(), None])


# A box clip splits the cells it cuts, and clips a point cloud through its vertices
assert_types(poly().clip_box(), pv.PolyData)
assert_types(pointset().clip_box(), pv.PointSet)
assert_types(image().clip_box(), pv.UnstructuredGrid)
assert_types(multiblock().clip_box(), pv.MultiBlock)

# A declared block type follows the filter through
assert_types(multiblock_poly().clip_box(), pv.MultiBlock[pv.PolyData])
assert_types(multiblock_image().clip_box(), pv.MultiBlock[pv.UnstructuredGrid])
assert_types(multiblock_pointset().clip_box(), pv.MultiBlock[pv.PointSet])
assert_types(multiblock_unstructured().clip_box(), pv.MultiBlock[pv.UnstructuredGrid])

# An empty block survives the filter
assert_types(multiblock_optional_poly().clip_box(), pv.MultiBlock[pv.PolyData | None])
assert_types(multiblock_optional_image().clip_box(), pv.MultiBlock[pv.UnstructuredGrid | None])
assert_types(multiblock_optional_pointset().clip_box(), pv.MultiBlock[pv.PointSet | None])
