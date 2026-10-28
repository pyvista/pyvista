"""Typing cases for :meth:`pyvista.DataObjectFilters.slice_orthogonal`."""

from __future__ import annotations

from typing import Never

from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import pointset


def poly() -> pv.PolyData:
    """Return a sphere."""
    return pv.Sphere(theta_resolution=8, phi_resolution=8)


def image() -> pv.ImageData:
    """Return a small uniform grid."""
    return pv.ImageData(dimensions=(5, 5, 5), spacing=(0.25, 0.25, 0.25), origin=(-0.5, -0.5, -0.5))


def multiblock() -> pv.MultiBlock:
    """Return a composite of two meshes."""
    return pv.MultiBlock([poly(), image()])


# The slices are collected into a composite, whatever went in
assert_types(image().slice_orthogonal(), pv.MultiBlock)
assert_types(multiblock().slice_orthogonal(), pv.MultiBlock)

# A point cloud cannot be reduced to a surface
assert_types(pointset().slice_orthogonal(), Never)
