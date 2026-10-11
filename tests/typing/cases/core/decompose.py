"""Typing cases for :meth:`pyvista.Transform.decompose`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

import pyvista as pv

_Five = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]

assert_types(pv.Transform().rotate_z(30.0).decompose(), _Five)
assert_types(pv.Transform().rotate_z(30.0).decompose(homogeneous=True), _Five)
