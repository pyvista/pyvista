"""Typing cases for :meth:`pyvista.UnstructuredGridFilters.subdivide_tetra`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import rectilinear


def tetra_grid() -> pv.UnstructuredGrid:
    """Return a grid made only of tetrahedra."""
    return rectilinear().to_tetrahedra()


assert_types(tetra_grid().subdivide_tetra(), pv.UnstructuredGrid)
