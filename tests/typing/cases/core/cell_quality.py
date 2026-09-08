"""Typing cases for :meth:`pyvista.DataObjectFilters.cell_quality`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv


def a_grid() -> pv.ImageData:
    """Return a small grid."""
    return pv.ImageData(dimensions=(3, 3, 3))


def a_multiblock() -> pv.MultiBlock:
    """Return a `MultiBlock` holding one mesh."""
    return pv.MultiBlock([pv.Sphere()])


assert_types(pv.Sphere().cell_quality(), pv.PolyData)
assert_types(a_grid().cell_quality(), pv.ImageData)
assert_types(a_multiblock().cell_quality(), pv.MultiBlock)
