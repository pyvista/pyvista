"""Typing cases for :meth:`pyvista.DataSetFilters.compute_boundary_mesh_quality`."""

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
    'pointset().compute_boundary_mesh_quality()': 'a `PointSet` has no cells, so the call raises',
}


assert_types(poly().compute_boundary_mesh_quality(), pv.PolyData)
assert_types(image().compute_boundary_mesh_quality(), pv.PolyData)
assert_types(rectilinear().compute_boundary_mesh_quality(), pv.PolyData)
assert_types(structured().compute_boundary_mesh_quality(), pv.PolyData)
assert_types(unstructured().compute_boundary_mesh_quality(), pv.PolyData)
assert_types(explicit_structured().compute_boundary_mesh_quality(), pv.PolyData)

assert_types(pointset().compute_boundary_mesh_quality(), Never)  # pragma: no cover
