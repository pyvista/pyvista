"""Typing cases for :meth:`pyvista.MultiBlock.append`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from pyvista import MultiBlock
from pyvista import PolyData


def multi() -> MultiBlock:
    """Return a `MultiBlock` with no declared block type."""
    return pv.MultiBlock([pv.PolyData()])


def polys() -> MultiBlock[PolyData]:
    """Return a `MultiBlock` declared to hold only `PolyData`."""
    return pv.MultiBlock([pv.PolyData()])


def nested() -> MultiBlock[MultiBlock[PolyData]]:
    """Return a `MultiBlock` declared to hold only composites of `PolyData`."""
    return pv.MultiBlock([polys()])


# Any leaf may be appended when no block type is declared
assert_types(multi().append(pv.PolyData()), None)
assert_types(multi().append(pv.ImageData()), None)
assert_types(multi().append(pv.MultiBlock()), None)
assert_types(multi().append(None), None)

# A declared block type accepts its own blocks
assert_types(polys().append(pv.PolyData()), None)
assert_types(polys().append(pv.Sphere()), None)

# A composite is a valid block whatever its own blocks are declared to be
assert_types(multi().append(polys()), None)
assert_types(multi().append(nested()), None)
