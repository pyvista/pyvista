"""Typing cases for :attr:`pyvista.Camera.extrinsic_matrix`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

import pyvista as pv

assert_types(pv.Camera().extrinsic_matrix, NDArray[np.float64])
