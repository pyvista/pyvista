"""Typing cases for :func:`pyvista.spherical_to_cartesian`."""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv
from pyvista.core._typing_core import NumpyArray

_Cartesian = tuple[NumpyArray[float], NumpyArray[float], NumpyArray[float]]


def some_array() -> NumpyArray[float]:
    """Return spherical coordinates as an array."""
    return np.array([1.0, 2.0, 3.0])


def some_list() -> list[float]:
    """Return spherical coordinates as a list."""
    return [1.0, 2.0, 3.0]


assert_types(pv.spherical_to_cartesian(some_array(), some_array(), some_array()), _Cartesian)
assert_types(pv.spherical_to_cartesian(some_list(), some_list(), some_list()), _Cartesian)
