"""Typing cases for :meth:`pyvista.PointSet.explode`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv

assert_types(pv.PointSet(pv.Sphere().points).explode(), pv.PointSet)
assert_types(pv.PointSet(pv.Sphere().points).explode(factor=0.5), pv.PointSet)
