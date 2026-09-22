"""Typing cases for :meth:`pyvista.DataSetFilters.remove_unused_points`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv

assert_types(pv.Sphere().remove_unused_points(), pv.PolyData)
assert_types(pv.Sphere().cast_to_unstructured_grid().remove_unused_points(), pv.UnstructuredGrid)
