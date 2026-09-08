"""Typing cases for :func:`pyvista.fit_plane_to_points`."""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv
from pyvista.core._typing_core import NumpyArray

_Result = pv.PolyData | tuple[pv.PolyData, NumpyArray[np.floating], NumpyArray[np.floating]]


def some_points() -> NumpyArray[float]:
    """Return the points of a sphere."""
    return pv.Sphere().points


# The metadata carries the points' own float width, and an axis name is a valid normal
assert_types(pv.fit_plane_to_points(some_points()), _Result)
assert_types(pv.fit_plane_to_points(some_points(), return_meta=True), _Result)
assert_types(pv.fit_plane_to_points(some_points(), init_normal='z'), _Result)
assert_types(pv.fit_plane_to_points(some_points(), init_normal=(0.0, 0.0, 1.0)), _Result)
