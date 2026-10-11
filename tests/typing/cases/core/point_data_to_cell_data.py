"""Typing cases for :meth:`pyvista.DataObjectFilters.point_data_to_cell_data`."""

from __future__ import annotations

from type_assert import assert_types
from typing_extensions import Never

import pyvista as pv
from tests.typing.meshes import explicit_structured
from tests.typing.meshes import image
from tests.typing.meshes import multiblock
from tests.typing.meshes import multiblock_poly
from tests.typing.meshes import pointset
from tests.typing.meshes import poly
from tests.typing.meshes import rectilinear
from tests.typing.meshes import structured
from tests.typing.meshes import unstructured

# Every dataset gives back its own class
assert_types(poly().point_data_to_cell_data(), pv.PolyData)
assert_types(image().point_data_to_cell_data(), pv.ImageData)
assert_types(rectilinear().point_data_to_cell_data(), pv.RectilinearGrid)
assert_types(structured().point_data_to_cell_data(), pv.StructuredGrid)
assert_types(unstructured().point_data_to_cell_data(), pv.UnstructuredGrid)
assert_types(explicit_structured().point_data_to_cell_data(), pv.ExplicitStructuredGrid)
assert_types(multiblock().point_data_to_cell_data(), pv.MultiBlock)

# A declared block type survives the filter
assert_types(multiblock_poly().point_data_to_cell_data(), pv.MultiBlock[pv.PolyData])

assert_types(pointset().point_data_to_cell_data(), Never)
