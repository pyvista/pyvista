"""Typing cases for :attr:`pyvista.Camera.intrinsic_matrix`."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from type_assert import assert_types

import pyvista as pv

assert_types(pv.Plotter().camera.intrinsic_matrix, npt.NDArray[np.float64])
