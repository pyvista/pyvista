"""Typing cases for :meth:`pyvista.DataObjectFilters.compute_cell_sizes`."""

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
    'pointset().compute_cell_sizes()': 'a `PointSet` has no cells, so the call raises',
}


# Every dataset gives back its own class
assert_types(poly().compute_cell_sizes(), pv.PolyData)
assert_types(image().compute_cell_sizes(), pv.ImageData)
assert_types(rectilinear().compute_cell_sizes(), pv.RectilinearGrid)
assert_types(structured().compute_cell_sizes(), pv.StructuredGrid)
assert_types(unstructured().compute_cell_sizes(), pv.UnstructuredGrid)
assert_types(explicit_structured().compute_cell_sizes(), pv.ExplicitStructuredGrid)
assert_types(multiblock().compute_cell_sizes(), pv.MultiBlock)

assert_types(pointset().compute_cell_sizes(), Never)
