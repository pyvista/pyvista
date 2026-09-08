"""Typing cases for :meth:`pyvista.MultiBlock.__setitem__`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from pyvista import MultiBlock


def multi() -> MultiBlock:
    """Return a named `MultiBlock` holding a mesh and a nested block."""
    return pv.MultiBlock({'mesh': pv.PolyData(), 'nested': pv.MultiBlock([pv.PolyData()])})


assert_types(multi().__setitem__(0, pv.PolyData()), None)
assert_types(multi().__setitem__(0, pv.MultiBlock()), None)
assert_types(multi().__setitem__(0, None), None)
assert_types(multi().__setitem__('mesh', pv.PolyData()), None)
assert_types(multi().__setitem__(slice(0, 1), [pv.PolyData()]), None)
