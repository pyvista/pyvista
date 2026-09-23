"""Typing cases for :meth:`pyvista.DataSetFilters.remove_cells`."""

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
    'pointset().remove_cells([0])': 'a `PointSet` has no cells, so the call raises',
}


assert_types(poly().remove_cells([0]), pv.PolyData)
assert_types(image().remove_cells([0]), pv.UnstructuredGrid)
assert_types(rectilinear().remove_cells([0]), pv.UnstructuredGrid)
assert_types(structured().remove_cells([0]), pv.UnstructuredGrid)
assert_types(unstructured().remove_cells([0]), pv.UnstructuredGrid)
assert_types(explicit_structured().remove_cells([0]), pv.UnstructuredGrid)

assert_types(pointset().remove_cells([0]), Never)  # pragma: no cover
