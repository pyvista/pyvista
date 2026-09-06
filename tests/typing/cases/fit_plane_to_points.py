"""Typing cases for :func:`pyvista.fit_plane_to_points`.

`return_meta` picks which half of the union comes back; the signature is not overloaded.
"""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv
from pyvista.core._typing_core import NumpyArray

_Fitted = pv.PolyData | tuple[pv.PolyData, NumpyArray[np.floating], NumpyArray[np.floating]]


def some_points() -> NumpyArray[float]:
    """Return points with a distinct variance along each axis."""
    return np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 0.5]])


assert_types(pv.fit_plane_to_points(some_points()), _Fitted)
assert_types(pv.fit_plane_to_points(some_points(), return_meta=False), _Fitted)
assert_types(pv.fit_plane_to_points(some_points(), return_meta=True), _Fitted)
assert_types(pv.fit_plane_to_points(some_points(), init_normal='-z'), _Fitted)
assert_types(pv.fit_plane_to_points(some_points(), init_normal=(0.0, 0.0, 1.0)), _Fitted)
