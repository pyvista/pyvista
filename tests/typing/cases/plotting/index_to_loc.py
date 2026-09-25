"""Typing cases for :meth:`pyvista.plotting.renderers.Renderers.index_to_loc`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

import pyvista as pv

assert_types(pv.Plotter(shape=(2, 2)).renderers.index_to_loc(1), NDArray[np.intp] | np.intp)
assert_types(pv.Plotter(shape=(1, 2)).renderers.index_to_loc(1), NDArray[np.intp] | np.intp)
