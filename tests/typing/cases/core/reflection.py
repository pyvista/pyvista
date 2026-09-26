"""Typing cases for :func:`pyvista.core.utilities.transformations.reflection`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

from pyvista.core.utilities.transformations import reflection


def float32_axis() -> NDArray[np.float32]:
    """Return a float32 axis."""
    return np.array([0, 0, 1], dtype=np.float32)


assert_types(reflection((0, 0, 1)), NDArray[np.float64])
assert_types(reflection((0.0, 0.0, 1.0), point=(1.0, 0.0, 0.0)), NDArray[np.float64])
assert_types(reflection(float32_axis()), NDArray[np.float64])
