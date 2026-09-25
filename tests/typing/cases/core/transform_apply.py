"""Typing cases for :meth:`pyvista.Transform.apply`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

import pyvista as pv


def a_transform() -> pv.Transform:
    """Return a transform with a translation composed into it."""
    return pv.Transform().translate((1.0, 2.0, 3.0))


def a_multiblock() -> pv.MultiBlock:
    """Return a `MultiBlock` holding one mesh."""
    return pv.MultiBlock([pv.Sphere()])


# Arrays
def a_polydata_multiblock() -> pv.MultiBlock[pv.PolyData]:
    """Return a `MultiBlock` declared to hold only `PolyData`."""
    return pv.MultiBlock([pv.Sphere()])


assert_types(a_transform().apply(np.zeros((4, 3))), NDArray[float])
assert_types(a_transform().apply([(0.0, 0.0, 0.0)]), NDArray[float])
assert_types(a_transform().apply((0.0, 0.0, 0.0)), NDArray[float])
assert_types(a_transform().apply(np.zeros((4, 3)), 'points'), NDArray[float])
assert_types(a_transform().apply(np.zeros((4, 3)), 'vectors'), NDArray[float])
assert_types(a_transform().apply(np.zeros((4, 3)), None), NDArray[float])

# Datasets keep their own type
assert_types(a_transform().apply(pv.Sphere()), pv.PolyData)
assert_types(a_transform().apply(pv.ImageData(dimensions=(2, 2, 2))), pv.ImageData)
assert_types(a_transform().apply(a_multiblock()), pv.MultiBlock)
assert_types(a_transform().apply(pv.Sphere(), 'active_vectors'), pv.PolyData)
assert_types(a_transform().apply(pv.Sphere(), 'all_vectors'), pv.PolyData)
assert_types(a_transform().apply(pv.Sphere(), inverse=True), pv.PolyData)

# Actors widen to the base class the overload names
assert_types(a_transform().apply(pv.Actor()), pv.Prop3D)
assert_types(a_transform().apply(pv.Actor(), 'replace'), pv.Prop3D)
assert_types(a_transform().apply(pv.Actor(), 'pre-multiply'), pv.Prop3D)
assert_types(a_transform().apply(pv.Actor(), 'post-multiply'), pv.Prop3D)

# A declared block type survives the transform
assert_types(pv.Transform().apply(a_polydata_multiblock()), pv.MultiBlock[pv.PolyData])
