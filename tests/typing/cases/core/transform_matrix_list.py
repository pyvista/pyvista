"""Typing cases for :attr:`pyvista.Transform.matrix_list`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

import pyvista as pv

assert_types(pv.Transform().scale(2.0).translate(1.0, 0.0, 0.0).matrix_list, list[NDArray[np.float64]])
