"""Typing cases for :meth:`pyvista.DataSetFilters.shrink`."""

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
    'pointset().shrink()': 'a `PointSet` has no cells, so the call raises',
}


assert_types(poly().shrink(), pv.PolyData)
assert_types(image().shrink(), pv.UnstructuredGrid)
assert_types(rectilinear().shrink(), pv.UnstructuredGrid)
assert_types(structured().shrink(), pv.UnstructuredGrid)
assert_types(unstructured().shrink(), pv.UnstructuredGrid)
assert_types(explicit_structured().shrink(), pv.UnstructuredGrid)

assert_types(pointset().shrink(), Never)  # pragma: no cover
