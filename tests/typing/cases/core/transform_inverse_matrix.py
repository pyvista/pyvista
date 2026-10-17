"""Typing cases for :attr:`pyvista.Transform.inverse_matrix`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

import pyvista as pv

assert_types(pv.Transform().scale(2.0).inverse_matrix, NDArray[np.float64])
