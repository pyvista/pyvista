"""Typing cases for :meth:`pyvista.PointSet.reconstruct_surface`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv

assert_types(pv.PointSet(pv.Sphere().points).reconstruct_surface(), pv.PolyData)
