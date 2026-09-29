"""Typing cases for :meth:`pyvista.Plotter.add_arrows`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from type_assert import assert_types

import pyvista as pv

if TYPE_CHECKING:
    from numpy.typing import NDArray


def float32_centers() -> NDArray[np.float32]:
    """Return float32 arrow centers."""
    return np.zeros((2, 3), dtype=np.float32)


def int32_directions() -> NDArray[np.int32]:
    """Return int32 arrow directions."""
    return np.ones((2, 3), dtype=np.int32)


assert_types(pv.Plotter().add_arrows(np.zeros((2, 3)), np.ones((2, 3))), pv.Actor)
assert_types(pv.Plotter().add_arrows(float32_centers(), int32_directions()), pv.Actor)
