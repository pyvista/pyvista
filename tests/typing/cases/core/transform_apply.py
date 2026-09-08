"""Typing cases for :meth:`pyvista.Transform.apply`."""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv
from pyvista.core._typing_core import NumpyArray


def a_transform() -> pv.Transform:
    """Return a transform with a translation composed into it."""
    return pv.Transform().translate((1.0, 2.0, 3.0))


def a_multiblock() -> pv.MultiBlock:
    """Return a `MultiBlock` holding one mesh."""
    return pv.MultiBlock([pv.Sphere()])


# Arrays
assert_types(a_transform().apply(np.zeros((4, 3))), NumpyArray[float])
assert_types(a_transform().apply([(0.0, 0.0, 0.0)]), NumpyArray[float])
assert_types(a_transform().apply((0.0, 0.0, 0.0)), NumpyArray[float])
assert_types(a_transform().apply(np.zeros((4, 3)), 'points'), NumpyArray[float])
assert_types(a_transform().apply(np.zeros((4, 3)), 'vectors'), NumpyArray[float])
assert_types(a_transform().apply(np.zeros((4, 3)), None), NumpyArray[float])

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
