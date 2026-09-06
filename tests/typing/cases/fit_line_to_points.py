"""Typing cases for :func:`pyvista.fit_line_to_points`.

`return_meta` picks which half of the union comes back; the signature is not overloaded.
"""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv
from pyvista.core._typing_core import NumpyArray

_Fitted = pv.PolyData | tuple[pv.PolyData, float, NumpyArray[float]]


def some_points() -> NumpyArray[float]:
    """Return points spread along one direction."""
    return np.array([[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [2.0, 0.0, 0.1], [3.0, 0.1, 0.0]])


assert_types(pv.fit_line_to_points(some_points()), _Fitted)
assert_types(pv.fit_line_to_points(some_points(), return_meta=False), _Fitted)
assert_types(pv.fit_line_to_points(some_points(), return_meta=True), _Fitted)
assert_types(pv.fit_line_to_points(some_points(), init_direction='x'), _Fitted)
assert_types(pv.fit_line_to_points(some_points(), init_direction=(1.0, 0.0, 0.0)), _Fitted)
