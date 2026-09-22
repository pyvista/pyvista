"""Typing cases for :meth:`pyvista.CompositeFilters.outline_corners`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import multiblock


def a_flag() -> bool:
    """Return a flag typed only as ``bool``, so the catch-all overload applies."""
    return True


# One outline of the whole composite, or one per block
assert_types(multiblock().outline_corners(), pv.PolyData)
assert_types(multiblock().outline_corners(factor=0.5), pv.PolyData)
assert_types(multiblock().outline_corners(nested=False), pv.PolyData)
assert_types(multiblock().outline_corners(nested=True), pv.MultiBlock)

# The catch-all, reached only by a flag widened to `bool`
assert_types(multiblock().outline_corners(nested=a_flag()), pv.PolyData | pv.MultiBlock)
