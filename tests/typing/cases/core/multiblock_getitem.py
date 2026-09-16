"""Typing cases for :meth:`pyvista.MultiBlock.__getitem__`."""

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


def optional_polys() -> MultiBlock[PolyData | None]:
    """Return a `MultiBlock` whose blocks may be missing."""
    return pv.MultiBlock([pv.PolyData(), None])


def nested() -> MultiBlock[MultiBlock[PolyData]]:
    """Return a `MultiBlock` declared to hold only `MultiBlock` of `PolyData`."""
    return pv.MultiBlock([polys()])


def an_index() -> int:
    """Return an index typed only as ``int``."""
    return 0


# A block is a dataset, a nested `MultiBlock` of any block type, or nothing at all
assert_types(multi()[0], MultiBlock[Any] | DataSet | None)
assert_types(multi()['mesh'], MultiBlock[Any] | DataSet | None)
assert_types(multi()[an_index()], MultiBlock[Any] | DataSet | None)

# Slicing keeps the container
assert_types(multi()[0:1], MultiBlock)
assert_types(multi()[:], MultiBlock)

# A parameterized `MultiBlock` indexes to its own block type
assert_types(polys()[0], PolyData)
assert_types(polys()['mesh'], PolyData)
assert_types(polys()[an_index()], PolyData)
assert_types(optional_polys()[0], PolyData | None)
assert_types(nested()[0][0], PolyData)

# Slicing keeps the block type too
assert_types(polys()[0:1], MultiBlock[PolyData])
