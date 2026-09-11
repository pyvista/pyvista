"""Typing cases for :meth:`pyvista.DataSetFilters.decimate_boundary`."""

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
    'pointset().decimate_boundary()': 'a `PointSet` has no cells, so the call raises',
}


assert_types(poly().decimate_boundary(), pv.PolyData)
assert_types(image().decimate_boundary(), pv.PolyData)
assert_types(rectilinear().decimate_boundary(), pv.PolyData)
assert_types(structured().decimate_boundary(), pv.PolyData)
assert_types(unstructured().decimate_boundary(), pv.PolyData)
assert_types(explicit_structured().decimate_boundary(), pv.PolyData)

assert_types(pointset().decimate_boundary(), Never)  # pragma: no cover
