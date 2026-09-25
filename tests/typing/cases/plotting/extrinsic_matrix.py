"""Typing cases for :attr:`pyvista.Camera.extrinsic_matrix`."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from type_assert import assert_types

import pyvista as pv

assert_types(pv.Camera().extrinsic_matrix, npt.NDArray[np.float64])
