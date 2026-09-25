"""Typing cases for :meth:`pyvista.DataSet.find_containing_cell`."""

from __future__ import annotations

import numpy.typing as npt
from type_assert import assert_types

import pyvista as pv


def an_image() -> pv.ImageData:
    """Return a small grid to query."""
    return pv.ImageData(dimensions=(3, 3, 3))


SKIP_RUNTIME = {
    'an_image().find_containing_cell([(1.0, 1.0, 1.0), (0.5, 0.5, 0.5)])': 'the runtime checker does not accept an int64 array as `npt.NDArray[int]`',
}


assert_types(an_image().find_containing_cell((1.0, 1.0, 1.0)), int | npt.NDArray[int])
assert_types(an_image().find_containing_cell([(1.0, 1.0, 1.0), (0.5, 0.5, 0.5)]), int | npt.NDArray[int])  # pragma: no cover
