"""Typing cases for :func:`pyvista.cartesian_to_spherical`."""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv
from pyvista.core._typing_core import NumpyArray

_Spherical = tuple[NumpyArray[float], NumpyArray[float], NumpyArray[float]]


def a_coordinate() -> NumpyArray[float]:
    """Return one Cartesian component of a few points."""
    return np.array([0.0, 1.0, 2.0])


def a_coordinate_grid() -> NumpyArray[float]:
    """Return one Cartesian component of a grid of points."""
    return np.array([[0.0, 1.0], [2.0, 3.0]])


assert_types(pv.cartesian_to_spherical(a_coordinate(), a_coordinate(), a_coordinate()), _Spherical)
assert_types(pv.cartesian_to_spherical(x=a_coordinate(), y=a_coordinate(), z=a_coordinate()), _Spherical)
assert_types(pv.cartesian_to_spherical(a_coordinate_grid(), a_coordinate_grid(), a_coordinate_grid()), _Spherical)
