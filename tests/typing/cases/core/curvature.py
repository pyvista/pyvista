"""Typing cases for :meth:`pyvista.PolyDataFilters.curvature`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

from tests.typing.meshes import poly

assert_types(poly().curvature(), NDArray[np.float64])
assert_types(poly().curvature('gaussian'), NDArray[np.float64])
