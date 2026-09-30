"""Typing cases for :attr:`pyvista.plotting.picking.PickingComponent.picked_point`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

import pyvista as pv

if TYPE_CHECKING:
    from pyvista.plotting.picking import PickingComponent


def a_component() -> PickingComponent:
    """Return the picking component of a plotter."""
    return pv.Plotter().picking


assert_types(a_component().picked_point, NDArray[np.float64] | None)
