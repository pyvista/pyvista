"""Typing cases for :func:`pyvista.plotting.tools.normalize`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

from pyvista.plotting.tools import normalize

assert_types(normalize(np.arange(4)), NDArray[np.floating])
assert_types(normalize(np.arange(4, dtype=np.float32), minimum=0.0, maximum=3.0), NDArray[np.floating])
