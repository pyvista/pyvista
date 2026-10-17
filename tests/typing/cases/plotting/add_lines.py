"""Typing cases for :meth:`pyvista.Plotter.add_lines`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from type_assert import assert_types

import pyvista as pv

if TYPE_CHECKING:
    from numpy.typing import NDArray


def int64_lines() -> NDArray[np.int64]:
    """Return int64 line segment end points."""
    return np.array([[0, 1, 0], [1, 0, 0], [1, 1, 0], [2, 0, 0]], dtype=np.int64)


def float32_lines() -> NDArray[np.float32]:
    """Return float32 line segment end points."""
    return np.zeros((2, 3), dtype=np.float32)


assert_types(pv.Plotter().add_lines(int64_lines()), pv.Actor)
assert_types(pv.Plotter().add_lines(float32_lines()), pv.Actor)
