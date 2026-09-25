"""Typing cases for :meth:`pyvista.LookupTable.__call__`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

import pyvista as pv

# A single value maps to one RGBA color
assert_types(pv.LookupTable()(0.0), tuple[float, float, float, float])

# A sequence, an array or a VTK array each map to an array of colors
assert_types(pv.LookupTable()([0.0, 1.0]), NDArray[float])
assert_types(pv.LookupTable()(np.linspace(0.0, 1.0, 4)), NDArray[float])
