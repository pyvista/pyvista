"""Typing cases for :meth:`pyvista.DataSet.find_containing_cell`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from pyvista.core._typing_core import NumpyArray


def an_image() -> pv.ImageData:
    """Return a small grid to query."""
    return pv.ImageData(dimensions=(3, 3, 3))


SKIP_RUNTIME = {
    'an_image().find_containing_cell([(1.0, 1.0, 1.0), (0.5, 0.5, 0.5)])': (
        'the runtime checker has no `npt_promote`, so an int64 array is not a `NumpyArray[int]`'
    ),
}

# fmt: off

assert_types(an_image().find_containing_cell((1.0, 1.0, 1.0)),                     int | NumpyArray[int])
assert_types(an_image().find_containing_cell([(1.0, 1.0, 1.0), (0.5, 0.5, 0.5)]),  int | NumpyArray[int])  # pragma: no cover
