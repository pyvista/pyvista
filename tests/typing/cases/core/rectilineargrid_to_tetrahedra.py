"""Typing cases for :meth:`pyvista.RectilinearGrid.to_tetrahedra`."""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import rectilinear


def mixed_grid() -> pv.RectilinearGrid:
    """Return a grid whose active cell scalars choose 5 or 12 tetrahedra per cell."""
    grid = rectilinear()
    grid.cell_data['mix'] = np.where(np.arange(grid.n_cells) % 2 == 0, 5, 12)
    return grid


# Always a grid of tetrahedra, however the cells are split
assert_types(rectilinear().to_tetrahedra(), pv.UnstructuredGrid)
assert_types(rectilinear().to_tetrahedra(tetra_per_cell=6), pv.UnstructuredGrid)
assert_types(rectilinear().to_tetrahedra(tetra_per_cell=12), pv.UnstructuredGrid)
assert_types(rectilinear().to_tetrahedra(pass_cell_ids=True, pass_data=True), pv.UnstructuredGrid)
assert_types(mixed_grid().to_tetrahedra(mixed=True), pv.UnstructuredGrid)
assert_types(mixed_grid().to_tetrahedra(mixed='mix'), pv.UnstructuredGrid)
