"""Typing cases for :meth:`pyvista.DataSetFilters.separate_cells`."""

from __future__ import annotations

from type_assert import assert_types
from typing_extensions import Never

import pyvista as pv
from tests.typing.meshes import explicit_structured
from tests.typing.meshes import image
from tests.typing.meshes import pointset
from tests.typing.meshes import poly
from tests.typing.meshes import rectilinear
from tests.typing.meshes import structured
from tests.typing.meshes import unstructured

SKIP_RUNTIME = {
    'pointset().separate_cells()': 'a `PointSet` has no cells, so the call raises',
}


assert_types(poly().separate_cells(), pv.PolyData)
assert_types(image().separate_cells(), pv.UnstructuredGrid)
assert_types(rectilinear().separate_cells(), pv.UnstructuredGrid)
assert_types(structured().separate_cells(), pv.UnstructuredGrid)
assert_types(unstructured().separate_cells(), pv.UnstructuredGrid)
assert_types(explicit_structured().separate_cells(), pv.UnstructuredGrid)

assert_types(pointset().separate_cells(), Never)  # pragma: no cover
