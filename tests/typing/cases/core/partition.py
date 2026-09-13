"""Typing cases for :meth:`pyvista.DataSetFilters.partition`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import explicit_structured
from tests.typing.meshes import image
from tests.typing.meshes import pointset
from tests.typing.meshes import poly
from tests.typing.meshes import rectilinear
from tests.typing.meshes import structured
from tests.typing.meshes import unstructured

assert_types(poly().partition(2), pv.MultiBlock)
assert_types(image().partition(2), pv.MultiBlock)
assert_types(rectilinear().partition(2), pv.MultiBlock)
assert_types(structured().partition(2), pv.MultiBlock)
assert_types(unstructured().partition(2), pv.MultiBlock)
assert_types(explicit_structured().partition(2), pv.MultiBlock)
assert_types(pointset().partition(2), pv.MultiBlock)

assert_types(poly().partition(2, as_composite=False), pv.UnstructuredGrid)
assert_types(image().partition(2, as_composite=False), pv.UnstructuredGrid)
assert_types(rectilinear().partition(2, as_composite=False), pv.UnstructuredGrid)
assert_types(structured().partition(2, as_composite=False), pv.UnstructuredGrid)
assert_types(unstructured().partition(2, as_composite=False), pv.UnstructuredGrid)
assert_types(explicit_structured().partition(2, as_composite=False), pv.UnstructuredGrid)
assert_types(pointset().partition(2, as_composite=False), pv.PointSet)
