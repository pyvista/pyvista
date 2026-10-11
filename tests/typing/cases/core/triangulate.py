"""Typing cases for :meth:`pyvista.DataObjectFilters.triangulate`."""

from __future__ import annotations

from type_assert import assert_types
from typing_extensions import Never

import pyvista as pv
from tests.typing.meshes import explicit_structured
from tests.typing.meshes import image
from tests.typing.meshes import multiblock
from tests.typing.meshes import multiblock_image
from tests.typing.meshes import multiblock_optional_image
from tests.typing.meshes import multiblock_optional_poly
from tests.typing.meshes import multiblock_poly
from tests.typing.meshes import pointset
from tests.typing.meshes import poly
from tests.typing.meshes import rectilinear
from tests.typing.meshes import structured
from tests.typing.meshes import unstructured

# A surface stays a surface; every other dataset is broken into linear cells
assert_types(poly().triangulate(), pv.PolyData)
assert_types(image().triangulate(), pv.UnstructuredGrid)
assert_types(rectilinear().triangulate(), pv.UnstructuredGrid)
assert_types(structured().triangulate(), pv.UnstructuredGrid)
assert_types(unstructured().triangulate(), pv.UnstructuredGrid)
assert_types(explicit_structured().triangulate(), pv.UnstructuredGrid)
assert_types(multiblock().triangulate(), pv.MultiBlock)

# A declared block type follows the filter through
assert_types(multiblock_poly().triangulate(), pv.MultiBlock[pv.PolyData])
assert_types(multiblock_image().triangulate(), pv.MultiBlock[pv.UnstructuredGrid])

# An empty block survives the filter
assert_types(multiblock_optional_poly().triangulate(), pv.MultiBlock[pv.PolyData | None])
assert_types(multiblock_optional_image().triangulate(), pv.MultiBlock[pv.UnstructuredGrid | None])


assert_types(pointset().triangulate(), Never)
