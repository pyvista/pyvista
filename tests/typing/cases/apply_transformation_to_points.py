"""Typing cases for :func:`pyvista.core.utilities.transformations.apply_transformation_to_points`."""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

from pyvista.core._typing_core import NumpyArray
from pyvista.core.utilities.transformations import apply_transformation_to_points


def a_transformation() -> NumpyArray[float]:
    """Return a 4x4 translation matrix."""
    matrix = np.eye(4)
    matrix[:3, 3] = [1.0, 2.0, 3.0]
    return matrix


def some_points() -> NumpyArray[float]:
    """Return points to transform."""
    return np.zeros((4, 3))


def a_flag() -> bool:
    """Return a flag typed only as ``bool``, so the catch-all overload applies."""
    return True


# fmt: off

assert_types(apply_transformation_to_points(a_transformation(), some_points()),                 NumpyArray[float])
assert_types(apply_transformation_to_points(a_transformation(), some_points(), inplace=False),  NumpyArray[float])
assert_types(apply_transformation_to_points(a_transformation(), some_points(), inplace=True),   None)

# The catch-all, reached only by a flag widened to `bool`
assert_types(apply_transformation_to_points(a_transformation(), some_points(), inplace=a_flag()), NumpyArray[float] | None)
