"""Typing cases for :meth:`pyvista.DataObjectFilters.ptc`."""

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
    'pointset().ptc()': 'a `PointSet` has no cells, so the call raises',
}


# Every dataset gives back its own class
assert_types(poly().ptc(), pv.PolyData)
assert_types(image().ptc(), pv.ImageData)
assert_types(rectilinear().ptc(), pv.RectilinearGrid)
assert_types(structured().ptc(), pv.StructuredGrid)
assert_types(unstructured().ptc(), pv.UnstructuredGrid)
assert_types(explicit_structured().ptc(), pv.ExplicitStructuredGrid)
assert_types(multiblock().ptc(), pv.MultiBlock)

assert_types(pointset().ptc(), Never)  # pragma: no cover
