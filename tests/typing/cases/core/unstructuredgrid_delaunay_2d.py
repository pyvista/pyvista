"""Typing cases for :meth:`pyvista.UnstructuredGridFilters.delaunay_2d`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv


def a_grid() -> pv.UnstructuredGrid:
    """Return a flat grid of points."""
    return pv.Plane(i_resolution=3, j_resolution=3).cast_to_unstructured_grid()


assert_types(a_grid().delaunay_2d(), pv.PolyData)
assert_types(a_grid().delaunay_2d(alpha=1.0), pv.PolyData)
