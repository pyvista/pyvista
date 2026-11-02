"""Typing cases for :meth:`pyvista.Color.__getitem__`."""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv

# A channel index or name gives one value
assert_types(pv.Color('red')[0], float)
assert_types(pv.Color('red')[np.int64(0)], float)
assert_types(pv.Color('red')['r'], float)

# A slice gives several
assert_types(pv.Color('red')[:3], tuple[float, ...])
