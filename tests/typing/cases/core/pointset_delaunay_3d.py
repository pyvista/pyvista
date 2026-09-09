"""Typing cases for :meth:`pyvista.PointSet.delaunay_3d`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv

assert_types(pv.PointSet(pv.Sphere().points).delaunay_3d(), pv.UnstructuredGrid)
