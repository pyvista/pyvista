"""Typing cases for :meth:`pyvista.DataObjectFilters.slice_implicit`."""

from __future__ import annotations

from type_assert import assert_types
from typing_extensions import Never

import pyvista as pv
from pyvista import _vtk
from pyvista.core.utilities import generate_plane

SKIP_RUNTIME = {
    'multiblock_pointset().slice_implicit(a_plane())': 'a `PointSet` has no cells, so the call raises',
}


def poly() -> pv.PolyData:
    """Return a sphere."""
    return pv.Sphere(theta_resolution=8, phi_resolution=8)


def image() -> pv.ImageData:
    """Return a small uniform grid."""
    return pv.ImageData(dimensions=(5, 5, 5), spacing=(0.25, 0.25, 0.25), origin=(-0.5, -0.5, -0.5))


def multiblock_poly() -> pv.MultiBlock[pv.PolyData]:
    """Return a composite declared to hold only `PolyData`."""
    return pv.MultiBlock([poly()])


def multiblock_image() -> pv.MultiBlock[pv.ImageData]:
    """Return a composite declared to hold only `ImageData`."""
    return pv.MultiBlock([image()])


def multiblock() -> pv.MultiBlock:
    """Return a composite of two meshes."""
    return pv.MultiBlock([poly(), image()])


def a_plane() -> _vtk.vtkPlane:
    """Return a plane through the origin."""
    return generate_plane((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def multiblock_optional_poly() -> pv.MultiBlock[pv.PolyData | None]:
    """Return a composite whose blocks may be missing."""
    return pv.MultiBlock([poly(), None])


def pointset() -> pv.PointSet:  # pragma: no cover
    """Return a point cloud."""
    return pv.PointSet(poly().points)


def multiblock_pointset() -> pv.MultiBlock[pv.PointSet]:  # pragma: no cover
    """Return a composite declared to hold only `PointSet`."""
    return pv.MultiBlock([pointset()])


# Slicing reduces any dataset to a surface, and a composite stays a composite
assert_types(image().slice_implicit(a_plane()), pv.PolyData)
assert_types(multiblock().slice_implicit(a_plane()), pv.MultiBlock)

# A declared block type follows the filter through
assert_types(multiblock_poly().slice_implicit(a_plane()), pv.MultiBlock[pv.PolyData])
assert_types(multiblock_image().slice_implicit(a_plane()), pv.MultiBlock[pv.PolyData])

# An empty block survives the filter
assert_types(multiblock_optional_poly().slice_implicit(a_plane()), pv.MultiBlock[pv.PolyData | None])

# A composite of point clouds cannot be reduced to a surface
assert_types(multiblock_pointset().slice_implicit(a_plane()), Never)  # pragma: no cover
