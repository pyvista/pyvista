"""Typing cases for :func:`pyvista.fit_line_to_points`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from pyvista.core._typing_core import NumpyArray

_Result = pv.PolyData | tuple[pv.PolyData, float, NumpyArray[float]]


def some_points() -> NumpyArray[float]:
    """Return the points of a sphere."""
    return pv.Sphere().points


# An axis name is a valid direction
assert_types(pv.fit_line_to_points(some_points()), _Result)
assert_types(pv.fit_line_to_points(some_points(), return_meta=True), _Result)
assert_types(pv.fit_line_to_points(some_points(), init_direction='z'), _Result)
assert_types(pv.fit_line_to_points(some_points(), init_direction=(0.0, 0.0, 1.0)), _Result)
