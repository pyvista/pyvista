"""Typing cases for :func:`pyvista.core.utilities.transformations.axis_angle_rotation`."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from type_assert import assert_types

from pyvista.core.utilities.transformations import axis_angle_rotation

assert_types(axis_angle_rotation((0, 0, 1), 30), npt.NDArray[np.float64])
assert_types(axis_angle_rotation((0.0, 0.0, 1.0), 0.5, point=(1.0, 0.0, 0.0), deg=False), npt.NDArray[np.float64])
assert_types(axis_angle_rotation(np.array([0, 0, 1], dtype=np.float32), 30), npt.NDArray[np.float64])
