"""Typing cases for :meth:`pyvista.PolyDataFilters.contour_banded`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv


def a_surface() -> pv.PolyData:
    """Return a sphere with point scalars to contour."""
    sphere = pv.Sphere()
    sphere['data'] = sphere.points[:, 2]
    return sphere


def a_flag() -> bool:
    """Return a flag typed only as ``bool``, so the catch-all overload applies."""
    return True


assert_types(a_surface().contour_banded(5), tuple[pv.PolyData, pv.PolyData])
assert_types(a_surface().contour_banded(5, generate_contour_edges=True), tuple[pv.PolyData, pv.PolyData])

assert_types(a_surface().contour_banded(5, generate_contour_edges=False), pv.PolyData)

# The catch-all, reached only by a flag widened to `bool`
assert_types(a_surface().contour_banded(5, generate_contour_edges=a_flag()), pv.PolyData | tuple[pv.PolyData, pv.PolyData])
