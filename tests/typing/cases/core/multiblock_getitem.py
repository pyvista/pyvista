"""Typing cases for :meth:`pyvista.MultiBlock.__getitem__`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from pyvista import DataSet
from pyvista import MultiBlock


def multi() -> MultiBlock:
    """Return a named `MultiBlock` holding a mesh and a nested block."""
    return pv.MultiBlock({'mesh': pv.PolyData(), 'nested': pv.MultiBlock([pv.PolyData()])})


def an_index() -> int:
    """Return an index typed only as ``int``."""
    return 0


# A block is a dataset, a nested `MultiBlock`, or nothing at all
assert_types(multi()[0], MultiBlock | DataSet | None)
assert_types(multi()['mesh'], MultiBlock | DataSet | None)
assert_types(multi()[an_index()], MultiBlock | DataSet | None)

# Slicing keeps the container
assert_types(multi()[0:1], MultiBlock)
assert_types(multi()[:], MultiBlock)
