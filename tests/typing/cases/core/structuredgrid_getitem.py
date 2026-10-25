"""Typing cases for :meth:`pyvista.StructuredGrid.__getitem__`."""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv
from pyvista import pyvista_ndarray


def a_grid() -> pv.StructuredGrid:
    """Return a grid carrying one point array and one cell array."""
    axis = np.arange(3, dtype=float)
    grid = pv.StructuredGrid(*np.meshgrid(axis, axis, axis, indexing='ij'))
    grid.point_data['data'] = np.arange(grid.n_points)
    grid.cell_data['data'] = np.arange(grid.n_cells)
    return grid


# A name reads an array
assert_types(a_grid()['data'], pyvista_ndarray)
assert_types(a_grid()['data', 'point'], pyvista_ndarray)
assert_types(a_grid()['data', 'cell'], pyvista_ndarray)

# An index reads a subset of the grid
assert_types(a_grid()[:, :, 1], pv.StructuredGrid)
assert_types(a_grid()[0:2, 0:2, 0:2], pv.StructuredGrid)
