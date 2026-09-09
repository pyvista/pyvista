"""Typing cases for :meth:`pyvista.DataObjectFilters.triangulate`."""

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
    'pointset().triangulate()': 'a `PointSet` has no cells, so the call raises',
}


# A surface stays a surface; every other dataset is broken into linear cells
assert_types(poly().triangulate(), pv.PolyData)
assert_types(image().triangulate(), pv.UnstructuredGrid)
assert_types(rectilinear().triangulate(), pv.UnstructuredGrid)
assert_types(structured().triangulate(), pv.UnstructuredGrid)
assert_types(unstructured().triangulate(), pv.UnstructuredGrid)
assert_types(explicit_structured().triangulate(), pv.UnstructuredGrid)
assert_types(multiblock().triangulate(), pv.MultiBlock)

# A `PointSet` is rejected outright, so the call never returns
assert_types(pointset().triangulate(), Never)
