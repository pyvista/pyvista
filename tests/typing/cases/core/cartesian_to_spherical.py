"""Typing cases for :func:`pyvista.cartesian_to_spherical`."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from type_assert import assert_types

import pyvista as pv

_Spherical = tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[float]]


def a_coordinate() -> npt.NDArray[float]:
    """Return one Cartesian component of a few points."""
    return np.array([0.0, 1.0, 2.0])


def a_coordinate_grid() -> npt.NDArray[float]:
    """Return one Cartesian component of a grid of points."""
    return np.array([[0.0, 1.0], [2.0, 3.0]])


assert_types(pv.cartesian_to_spherical(a_coordinate(), a_coordinate(), a_coordinate()), _Spherical)
assert_types(pv.cartesian_to_spherical(x=a_coordinate(), y=a_coordinate(), z=a_coordinate()), _Spherical)
assert_types(pv.cartesian_to_spherical(a_coordinate_grid(), a_coordinate_grid(), a_coordinate_grid()), _Spherical)
