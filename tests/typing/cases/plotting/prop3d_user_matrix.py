"""Typing cases for :attr:`pyvista.Prop3D.user_matrix`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

import pyvista as pv

assert_types(pv.Actor().user_matrix, NDArray[np.float64])
assert_types(pv.AxesAssembly().user_matrix, NDArray[np.float64])
