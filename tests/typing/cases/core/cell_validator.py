"""Typing cases for :meth:`pyvista.DataObjectFilters.cell_validator`."""

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
    'pointset().cell_validator()': 'a `PointSet` has no cells, so the call raises',
}


# Every dataset gives back its own class
assert_types(poly().cell_validator(), pv.PolyData)
assert_types(image().cell_validator(), pv.ImageData)
assert_types(rectilinear().cell_validator(), pv.RectilinearGrid)
assert_types(structured().cell_validator(), pv.StructuredGrid)
assert_types(unstructured().cell_validator(), pv.UnstructuredGrid)
assert_types(explicit_structured().cell_validator(), pv.ExplicitStructuredGrid)
assert_types(multiblock().cell_validator(), pv.MultiBlock)

assert_types(pointset().cell_validator(), Never)
