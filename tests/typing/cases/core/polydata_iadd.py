"""Typing cases for :meth:`pyvista.PolyDataFilters.__iadd__`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import poly

# An in-place merge keeps this mesh, which is why it accepts polydata alone
mesh = poly()
mesh += poly()
assert_types(mesh, pv.PolyData)
