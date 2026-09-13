"""Typing cases for :meth:`pyvista.DataSetFilters.extract_cells_by_type`."""

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
    'pointset().extract_cells_by_type([pv.CellType.TRIANGLE, pv.CellType.VOXEL, pv.CellType.HEXAHEDRON])': 'a `PointSet` has no cells, so the call raises',
}


assert_types(poly().extract_cells_by_type([pv.CellType.TRIANGLE, pv.CellType.VOXEL, pv.CellType.HEXAHEDRON]), pv.PolyData)
assert_types(image().extract_cells_by_type([pv.CellType.TRIANGLE, pv.CellType.VOXEL, pv.CellType.HEXAHEDRON]), pv.ImageData)
assert_types(rectilinear().extract_cells_by_type([pv.CellType.TRIANGLE, pv.CellType.VOXEL, pv.CellType.HEXAHEDRON]), pv.RectilinearGrid)
assert_types(structured().extract_cells_by_type([pv.CellType.TRIANGLE, pv.CellType.VOXEL, pv.CellType.HEXAHEDRON]), pv.StructuredGrid)
assert_types(unstructured().extract_cells_by_type([pv.CellType.TRIANGLE, pv.CellType.VOXEL, pv.CellType.HEXAHEDRON]), pv.UnstructuredGrid)
assert_types(explicit_structured().extract_cells_by_type([pv.CellType.TRIANGLE, pv.CellType.VOXEL, pv.CellType.HEXAHEDRON]), pv.ExplicitStructuredGrid)

assert_types(pointset().extract_cells_by_type([pv.CellType.TRIANGLE, pv.CellType.VOXEL, pv.CellType.HEXAHEDRON]), Never)  # pragma: no cover
