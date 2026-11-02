"""Typing cases for :meth:`pyvista.MultiBlock.replace`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from pyvista import MultiBlock
from pyvista import PolyData


def multi() -> MultiBlock:
    """Return a named `MultiBlock` holding a mesh and a nested block."""
    return pv.MultiBlock({'mesh': pv.PolyData(), 'nested': pv.MultiBlock([pv.PolyData()])})


def polys() -> MultiBlock[PolyData]:
    """Return a named `MultiBlock` declared to hold only `PolyData`."""
    return pv.MultiBlock({'mesh': pv.PolyData()})


assert_types(multi().replace(0, pv.ImageData()), None)
assert_types(multi().replace('mesh', None), None)
assert_types(multi().replace((1, 0), pv.ImageData()), None)
assert_types(polys().replace(0, pv.PolyData()), None)
assert_types(polys().replace([0], pv.ImageData()), None)
