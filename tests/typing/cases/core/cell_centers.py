"""Typing cases for :meth:`pyvista.DataObjectFilters.cell_centers`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import explicit_structured
from tests.typing.meshes import image
from tests.typing.meshes import multiblock
from tests.typing.meshes import pointset
from tests.typing.meshes import poly
from tests.typing.meshes import rectilinear
from tests.typing.meshes import structured
from tests.typing.meshes import unstructured

# Cell centres come back as points, whatever went in
assert_types(poly().cell_centers(), pv.PolyData)
assert_types(image().cell_centers(), pv.PolyData)
assert_types(rectilinear().cell_centers(), pv.PolyData)
assert_types(structured().cell_centers(), pv.PolyData)
assert_types(unstructured().cell_centers(), pv.PolyData)
assert_types(explicit_structured().cell_centers(), pv.PolyData)
assert_types(pointset().cell_centers(), pv.PolyData)
assert_types(multiblock().cell_centers(), pv.MultiBlock)
