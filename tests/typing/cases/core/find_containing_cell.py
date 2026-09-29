"""Typing cases for :meth:`pyvista.DataSet.find_containing_cell`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

import pyvista as pv


def an_image() -> pv.ImageData:
    """Return a small grid to query."""
    return pv.ImageData(dimensions=(3, 3, 3))


assert_types(an_image().find_containing_cell((1.0, 1.0, 1.0)), int | NDArray[np.int_])
assert_types(an_image().find_containing_cell([(1.0, 1.0, 1.0), (0.5, 0.5, 0.5)]), int | NDArray[np.int_])
