"""Typing cases for :attr:`pyvista.Transform.inverse_matrix_list`."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from type_assert import assert_types

import pyvista as pv

assert_types(pv.Transform().scale(2.0).translate(1.0, 0.0, 0.0).inverse_matrix_list, list[npt.NDArray[np.float64]])
