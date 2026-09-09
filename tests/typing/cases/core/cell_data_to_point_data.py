"""Typing cases for :meth:`pyvista.DataObjectFilters.cell_data_to_point_data`."""

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
    'pointset().cell_data_to_point_data()': 'a `PointSet` has no cells, so the call raises',
}


# Every dataset gives back its own class
assert_types(poly().cell_data_to_point_data(), pv.PolyData)
assert_types(image().cell_data_to_point_data(), pv.ImageData)
assert_types(rectilinear().cell_data_to_point_data(), pv.RectilinearGrid)
assert_types(structured().cell_data_to_point_data(), pv.StructuredGrid)
assert_types(unstructured().cell_data_to_point_data(), pv.UnstructuredGrid)
assert_types(explicit_structured().cell_data_to_point_data(), pv.ExplicitStructuredGrid)
assert_types(multiblock().cell_data_to_point_data(), pv.MultiBlock)

# A `PointSet` is rejected outright, so the call never returns
assert_types(pointset().cell_data_to_point_data(), Never)
