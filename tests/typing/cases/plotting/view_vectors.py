"""Typing cases for :func:`pyvista.plotting.helpers.view_vectors`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

from pyvista.plotting.helpers import view_vectors

assert_types(view_vectors('xy'), tuple[NDArray[np.int_], NDArray[np.int_]])
assert_types(view_vectors('yz', negative=True), tuple[NDArray[np.int_], NDArray[np.int_]])
