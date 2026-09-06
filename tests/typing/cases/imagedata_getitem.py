"""Typing cases for :meth:`pyvista.ImageData.__getitem__`."""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv
from pyvista import pyvista_ndarray


def an_image() -> pv.ImageData:
    """Return a grid carrying one point array and one cell array."""
    image = pv.ImageData(dimensions=(3, 3, 3))
    image.point_data['data'] = np.arange(image.n_points)
    image.cell_data['data'] = np.arange(image.n_cells)
    return image


# fmt: off

# A name reads an array, an index reads a subset of the grid
assert_types(an_image()['data'],                       pv.ImageData | pyvista_ndarray)
assert_types(an_image()['data', 'point'],              pv.ImageData | pyvista_ndarray)
assert_types(an_image()['data', 'cell'],               pv.ImageData | pyvista_ndarray)
assert_types(an_image()[0, 0, 0],                      pv.ImageData | pyvista_ndarray)
