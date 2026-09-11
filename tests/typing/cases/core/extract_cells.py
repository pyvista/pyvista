"""Typing cases for :meth:`pyvista.DataSetFilters.extract_cells`."""

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
    'pointset().extract_cells([0])': 'a `PointSet` has no cells, so the call raises',
}


assert_types(poly().extract_cells([0]), pv.UnstructuredGrid)
assert_types(image().extract_cells([0]), pv.UnstructuredGrid)
assert_types(rectilinear().extract_cells([0]), pv.UnstructuredGrid)
assert_types(structured().extract_cells([0]), pv.UnstructuredGrid)
assert_types(unstructured().extract_cells([0]), pv.UnstructuredGrid)
assert_types(explicit_structured().extract_cells([0]), pv.UnstructuredGrid)

assert_types(pointset().extract_cells([0]), Never)  # pragma: no cover
