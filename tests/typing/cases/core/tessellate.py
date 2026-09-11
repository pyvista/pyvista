"""Typing cases for :meth:`pyvista.DataSetFilters.tessellate`."""

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
    'poly().tessellate()': 'the filter rejects a PolyData',
    'pointset().tessellate()': 'a `PointSet` has no cells, so the call raises',
}


assert_types(poly().tessellate(), pv.UnstructuredGrid)
assert_types(image().tessellate(), pv.UnstructuredGrid)
assert_types(rectilinear().tessellate(), pv.UnstructuredGrid)
assert_types(structured().tessellate(), pv.UnstructuredGrid)
assert_types(unstructured().tessellate(), pv.UnstructuredGrid)
assert_types(explicit_structured().tessellate(), pv.UnstructuredGrid)

assert_types(pointset().tessellate(), Never)  # pragma: no cover
