"""Typing cases for :attr:`pyvista.Plotter.pickpoint`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

import pyvista as pv

assert_types(pv.Plotter().pickpoint, NDArray[np.float64] | None)
