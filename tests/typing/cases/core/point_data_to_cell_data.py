"""Typing cases for :meth:`pyvista.DataObjectFilters.point_data_to_cell_data`."""

from __future__ import annotations

from type_assert import assert_types
from typing_extensions import Never

import pyvista as pv
from tests.typing.meshes import explicit_structured
from tests.typing.meshes import image
from tests.typing.meshes import multiblock
from tests.typing.meshes import pointset
from tests.typing.meshes import poly
from tests.typing.meshes import rectilinear
from tests.typing.meshes import structured
from tests.typing.meshes import unstructured

SKIP_RUNTIME = {
    'pointset().point_data_to_cell_data()': 'a `PointSet` has no cells, so the call raises',
}


# Every dataset gives back its own class
assert_types(poly().point_data_to_cell_data(), pv.PolyData)
assert_types(image().point_data_to_cell_data(), pv.ImageData)
assert_types(rectilinear().point_data_to_cell_data(), pv.RectilinearGrid)
assert_types(structured().point_data_to_cell_data(), pv.StructuredGrid)
assert_types(unstructured().point_data_to_cell_data(), pv.UnstructuredGrid)
assert_types(explicit_structured().point_data_to_cell_data(), pv.ExplicitStructuredGrid)
assert_types(multiblock().point_data_to_cell_data(), pv.MultiBlock)

# A `PointSet` is rejected outright, so the call never returns
assert_types(pointset().point_data_to_cell_data(), Never)
