"""Typing cases for :meth:`pyvista.CompositeFilters.outline`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import multiblock

# One mesh either way, whether the outline is of the whole composite or of each block
assert_types(multiblock().outline(), pv.PolyData)
assert_types(multiblock().outline(generate_faces=True), pv.PolyData)
assert_types(multiblock().outline(nested=True), pv.PolyData)
assert_types(multiblock().outline(nested=True, generate_faces=True), pv.PolyData)
