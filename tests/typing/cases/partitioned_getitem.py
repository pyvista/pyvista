"""Typing cases for :meth:`pyvista.PartitionedDataSet.__getitem__` and ``__setitem__``."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from pyvista import DataSet
from pyvista import PartitionedDataSet


def partitions() -> PartitionedDataSet:
    """Return a `PartitionedDataSet` holding two meshes."""
    return pv.PartitionedDataSet([pv.PolyData(), pv.PolyData()])


def an_index() -> int:
    """Return an index typed only as ``int``."""
    return 0


assert_types(partitions()[0], DataSet | None)
assert_types(partitions()[an_index()], DataSet | None)

assert_types(partitions()[0:1], PartitionedDataSet)
assert_types(partitions()[:], PartitionedDataSet)

assert_types(partitions().__setitem__(0, pv.PolyData()), None)
assert_types(partitions().__setitem__(0, None), None)
assert_types(partitions().__setitem__(slice(0, 1), [pv.PolyData()]), None)
