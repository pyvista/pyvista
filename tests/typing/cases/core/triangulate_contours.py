"""Typing cases for :meth:`pyvista.PolyDataFilters.triangulate_contours`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv


def some_contours() -> pv.PolyData:
    """Return a mesh made of lines."""
    return pv.Circle().extract_feature_edges()


assert_types(some_contours().triangulate_contours(), pv.PolyData)
