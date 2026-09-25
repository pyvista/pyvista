"""Typing cases for :attr:`pyvista.Transform.inverse_matrix`."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from type_assert import assert_types

import pyvista as pv

assert_types(pv.Transform().scale(2.0).inverse_matrix, npt.NDArray[np.float64])
