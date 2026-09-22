"""Typing cases for :meth:`pyvista.DataObjectFilters.slice_implicit`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from pyvista import _vtk
from pyvista.core.utilities import generate_plane


def poly() -> pv.PolyData:
    """Return a sphere."""
    return pv.Sphere(theta_resolution=8, phi_resolution=8)


def image() -> pv.ImageData:
    """Return a small uniform grid."""
    return pv.ImageData(dimensions=(5, 5, 5), spacing=(0.25, 0.25, 0.25), origin=(-0.5, -0.5, -0.5))


def multiblock() -> pv.MultiBlock:
    """Return a composite of two meshes."""
    return pv.MultiBlock([poly(), image()])


def a_plane() -> _vtk.vtkPlane:
    """Return a plane through the origin."""
    return generate_plane((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))


# Slicing reduces any dataset to a surface, and a composite stays a composite
assert_types(image().slice_implicit(a_plane()), pv.PolyData)
assert_types(multiblock().slice_implicit(a_plane()), pv.MultiBlock)
