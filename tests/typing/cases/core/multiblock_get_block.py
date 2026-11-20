"""Typing cases for :meth:`pyvista.MultiBlock.get_block`."""

from __future__ import annotations

from typing import Any

from type_assert import assert_types

import pyvista as pv
from pyvista import DataSet
from pyvista import MultiBlock
from pyvista import PolyData


def multi() -> MultiBlock:
    """Return a named `MultiBlock` holding a mesh and a nested block."""
    return pv.MultiBlock({'mesh': pv.PolyData(), 'nested': pv.MultiBlock([pv.PolyData()])})


def polys() -> MultiBlock[PolyData]:
    """Return a named `MultiBlock` declared to hold only `PolyData`."""
    return pv.MultiBlock({'mesh': pv.PolyData()})


# An index or name gives the block type
assert_types(multi().get_block(0), MultiBlock[Any] | DataSet | None)
assert_types(multi().get_block('mesh'), MultiBlock[Any] | DataSet | None)
assert_types(polys().get_block(0), PolyData)
assert_types(polys().get_block('mesh'), PolyData)

# A nested index can reach a block of any type
assert_types(multi().get_block((1, 0)), MultiBlock[Any] | DataSet | None)
assert_types(polys().get_block([0]), MultiBlock[Any] | DataSet | None)
