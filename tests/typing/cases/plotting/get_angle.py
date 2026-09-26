"""Typing cases for :func:`pyvista.plotting.affine_widget.get_angle`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from type_assert import assert_types

from pyvista.plotting.affine_widget import get_angle

if TYPE_CHECKING:
    from numpy.typing import NDArray


def int32_vector(index: int) -> NDArray[np.int32]:
    """Return an int32 unit vector along ``index``."""
    return np.eye(3, dtype=np.int32)[index]


assert_types(get_angle([1, 0, 0], [0, 1, 0]), float)
assert_types(get_angle(int32_vector(0), int32_vector(1)), float)
