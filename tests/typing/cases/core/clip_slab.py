"""Typing cases for :meth:`pyvista.DataObjectFilters.clip_slab`."""

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


def multiblock() -> pv.MultiBlock:
    """Return a composite of two meshes."""
    return pv.MultiBlock([poly(), image()])


# A slab clip follows the plane clip
assert_types(poly().clip_slab(thickness=0.2, normal='z'), pv.PolyData)
assert_types(pointset().clip_slab(thickness=0.2, normal='z'), pv.PointSet)
assert_types(image().clip_slab(thickness=0.2, normal='z'), pv.UnstructuredGrid)
assert_types(multiblock().clip_slab(thickness=0.2, normal='z'), pv.MultiBlock)
