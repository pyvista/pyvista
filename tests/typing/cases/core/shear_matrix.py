"""Typing cases for :attr:`pyvista.Transform.shear_matrix`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

import pyvista as pv

assert_types(pv.Transform().rotate_z(30.0).shear_matrix, NDArray[np.float64])
