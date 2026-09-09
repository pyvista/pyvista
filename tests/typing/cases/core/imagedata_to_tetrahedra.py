"""Typing cases for :meth:`pyvista.ImageData.to_tetrahedra`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv

assert_types(pv.ImageData(dimensions=(3, 3, 3)).to_tetrahedra(), pv.UnstructuredGrid)
assert_types(pv.ImageData(dimensions=(3, 3, 3)).to_tetrahedra(tetra_per_cell=6), pv.UnstructuredGrid)
