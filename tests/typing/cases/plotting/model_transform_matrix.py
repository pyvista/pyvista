"""Typing cases for :attr:`pyvista.Camera.model_transform_matrix`."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from type_assert import assert_types

import pyvista as pv

assert_types(pv.Camera().model_transform_matrix, npt.NDArray[np.float64])
