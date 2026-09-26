"""Typing cases for :meth:`pyvista.Texture.to_array`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pyvista_validation._typing._array_like import _Scalar
from type_assert import assert_types

import pyvista as pv

assert_types(pv.Texture(np.zeros((4, 4, 3), dtype=np.uint8)).to_array(), NDArray[_Scalar])
assert_types(pv.Texture(np.zeros((4, 4, 3), dtype=np.float32)).to_array(), NDArray[_Scalar])
