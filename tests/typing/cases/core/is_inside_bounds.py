"""Typing cases for :func:`pyvista.core.utilities.is_inside_bounds`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from type_assert import assert_types

from pyvista.core.utilities import is_inside_bounds
from tests.typing.meshes import poly

if TYPE_CHECKING:
    from pyvista.core._typing_core import NumpyArray


def a_point() -> NumpyArray[float]:
    """Return a point at the centre of the unit cube."""
    return np.array([0.5, 0.5, 0.5])


def some_bounds() -> NumpyArray[float]:
    """Return the bounds of the unit cube."""
    return np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])


# A scalar point, checked against one-dimensional bounds
assert_types(is_inside_bounds(0.5, (0.0, 1.0)), bool)
assert_types(is_inside_bounds(2, (0.0, 1.0)), bool)

# A sequence point
assert_types(is_inside_bounds((0.5, 0.5, 0.5), (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)), bool)
assert_types(is_inside_bounds([0.5, 0.5, 0.5], poly().bounds), bool)

# An array point
assert_types(is_inside_bounds(a_point(), some_bounds()), bool)
assert_types(is_inside_bounds(a_point(), poly().bounds), bool)
