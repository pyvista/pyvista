"""Typing cases for :meth:`pyvista.Transform.apply_to_points`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

import pyvista as pv

assert_types(pv.Transform().scale(2.0).apply_to_points(np.zeros((2, 3), dtype=np.float32)), NDArray[np.float64])
assert_types(pv.Transform().scale(2.0).apply_to_points([(0.0, 0.0, 0.0)]), NDArray[np.float64])
assert_types(pv.Transform().scale(2.0).apply_to_points(np.zeros((2, 3)), copy=False), NDArray[np.floating])
