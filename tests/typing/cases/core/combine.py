"""Typing cases for :meth:`pyvista.CompositeFilters.combine`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import multiblock

# Blocks of any class append into one grid
assert_types(multiblock().combine(), pv.UnstructuredGrid)
assert_types(multiblock().combine(merge_points=True), pv.UnstructuredGrid)
assert_types(multiblock().combine(merge_points=True, tolerance=1e-6), pv.UnstructuredGrid)
assert_types(pv.MultiBlock().combine(), pv.UnstructuredGrid)
