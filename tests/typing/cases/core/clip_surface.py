"""Typing cases for :meth:`pyvista.DataSetFilters.clip_surface`."""

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


def a_surface() -> pv.PolyData:
    """Return a closed surface enclosing the test meshes."""
    return pv.Sphere(radius=2.0, theta_resolution=8, phi_resolution=8)


# Clipping by a surface follows the input class
assert_types(poly().clip_surface(a_surface()), pv.PolyData)
assert_types(pointset().clip_surface(a_surface()), pv.PointSet)
assert_types(image().clip_surface(a_surface()), pv.UnstructuredGrid)
