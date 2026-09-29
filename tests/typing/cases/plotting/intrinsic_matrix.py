"""Typing cases for :attr:`pyvista.Camera.intrinsic_matrix`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

import pyvista as pv

assert_types(pv.Plotter().camera.intrinsic_matrix, NDArray[np.float64])
