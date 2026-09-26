"""Typing cases for :meth:`pyvista.PolyDataFilters.ray_trace`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

from tests.typing.meshes import poly

assert_types(poly().ray_trace((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), tuple[NDArray[np.floating], NDArray[np.intp]])
assert_types(poly().ray_trace((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), first_point=True), tuple[NDArray[np.floating], NDArray[np.intp]])
