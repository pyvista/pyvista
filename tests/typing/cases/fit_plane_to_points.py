"""Typing cases for :func:`pyvista.fit_plane_to_points`.

`return_meta` picks which half of the union comes back, but the signature is not
overloaded, so a caller is told only that it is one of the two.
"""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv
from pyvista.core._typing_core import NumpyArray

_Fitted = pv.PolyData | tuple[pv.PolyData, NumpyArray[np.float32], NumpyArray[np.float32]]


def some_points() -> NumpyArray[float]:
    """Return points with a distinct variance along each axis."""
    return np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 0.5]])


# fmt: off

assert_types(pv.fit_plane_to_points(some_points()),                     _Fitted)
assert_types(pv.fit_plane_to_points(some_points(), return_meta=False),  _Fitted)
assert_types(pv.fit_plane_to_points(some_points(), return_meta=True),   _Fitted)
assert_types(pv.fit_plane_to_points(some_points(), resolution=4),       _Fitted)
