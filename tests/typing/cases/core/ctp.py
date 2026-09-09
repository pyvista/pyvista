"""Typing cases for :meth:`pyvista.DataObjectFilters.ctp`."""

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
    'pointset().ctp()': 'a `PointSet` has no cells, so the call raises',
}


# Every dataset gives back its own class
assert_types(poly().ctp(), pv.PolyData)
assert_types(image().ctp(), pv.ImageData)
assert_types(rectilinear().ctp(), pv.RectilinearGrid)
assert_types(structured().ctp(), pv.StructuredGrid)
assert_types(unstructured().ctp(), pv.UnstructuredGrid)
assert_types(explicit_structured().ctp(), pv.ExplicitStructuredGrid)
assert_types(multiblock().ctp(), pv.MultiBlock)

assert_types(pointset().ctp(), Never)
