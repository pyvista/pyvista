"""Typing cases for :class:`pyvista.Color`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from type_assert import assert_types

import pyvista as pv

if TYPE_CHECKING:
    from numpy.typing import NDArray


def uint8_rgb() -> NDArray[np.uint8]:
    """Return a uint8 RGB color."""
    return np.array([255, 0, 0], dtype=np.uint8)


def int64_rgb() -> NDArray[np.int64]:
    """Return an int64 RGB color."""
    return np.array([255, 0, 0], dtype=np.int64)


def float32_rgb() -> NDArray[np.float32]:
    """Return a float32 RGB color."""
    return np.array([1.0, 0.0, 0.0], dtype=np.float32)


assert_types(pv.Color(uint8_rgb()), pv.Color)
assert_types(pv.Color(int64_rgb()), pv.Color)
assert_types(pv.Color(float32_rgb()), pv.Color)
