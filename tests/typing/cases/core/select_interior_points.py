"""Typing cases for :meth:`pyvista.DataSetFilters.select_interior_points`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv


def a_grid() -> pv.ImageData:
    """Return a small grid."""
    return pv.ImageData(dimensions=(3, 3, 3))


assert_types(a_grid().select_interior_points(pv.Sphere()), pv.ImageData)
assert_types(pv.Sphere().select_interior_points(pv.Sphere()), pv.PolyData)
