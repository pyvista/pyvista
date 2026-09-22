"""Typing cases for :meth:`pyvista.PartitionedDataSet.__setitem__`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from pyvista import PartitionedDataSet


def partitions() -> PartitionedDataSet:
    """Return a `PartitionedDataSet` holding two meshes."""
    return pv.PartitionedDataSet([pv.PolyData(), pv.PolyData()])


assert_types(partitions().__setitem__(0, pv.PolyData()), None)
assert_types(partitions().__setitem__(0, None), None)
assert_types(partitions().__setitem__(slice(0, 1), [pv.PolyData()]), None)
