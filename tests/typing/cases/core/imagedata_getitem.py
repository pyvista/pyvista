"""Typing cases for :meth:`pyvista.ImageData.__getitem__`."""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv
from pyvista import pyvista_ndarray

_Index = int | slice | tuple[int, int]


def an_image() -> pv.ImageData:
    """Return a grid carrying one point array and one cell array."""
    image = pv.ImageData(dimensions=(3, 3, 3))
    image.point_data['data'] = np.arange(image.n_points)
    image.cell_data['data'] = np.arange(image.n_cells)
    return image


def a_key() -> str | tuple[_Index, _Index, _Index]:
    """Return a key typed as everything the subscript accepts, so the union comes back."""
    return 'data'


# A name reads an array
assert_types(an_image()['data'], pyvista_ndarray)
assert_types(an_image()['data', 'point'], pyvista_ndarray)
assert_types(an_image()['data', 'cell'], pyvista_ndarray)

# An index reads a subset of the grid
assert_types(an_image()[0, 0, 0], pv.ImageData)
assert_types(an_image()[:, :, 1], pv.ImageData)
assert_types(an_image()[0:2, 0:2, 0:2], pv.ImageData)
assert_types(an_image()[(0, 1), :, 1], pv.ImageData)

# A key that could be either gives back either
assert_types(an_image()[a_key()], pv.ImageData | pyvista_ndarray)
