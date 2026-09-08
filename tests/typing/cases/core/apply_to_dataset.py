"""Typing cases for :meth:`pyvista.Transform.apply_to_dataset`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv


def a_grid() -> pv.ImageData:
    """Return a small grid."""
    return pv.ImageData(dimensions=(3, 3, 3))


def a_multiblock() -> pv.MultiBlock:
    """Return a `MultiBlock` holding one mesh."""
    return pv.MultiBlock([pv.Sphere()])


assert_types(pv.Transform().apply_to_dataset(pv.Sphere()), pv.PolyData)
assert_types(pv.Transform().apply_to_dataset(a_grid()), pv.ImageData)
assert_types(pv.Transform().apply_to_dataset(a_multiblock()), pv.MultiBlock)
