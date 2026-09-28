"""Typing cases for :meth:`pyvista.DataSetFilters.merge_points`."""

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

assert_types(poly().merge_points(), pv.PolyData)
assert_types(image().merge_points(), pv.UnstructuredGrid)
assert_types(rectilinear().merge_points(), pv.UnstructuredGrid)
assert_types(structured().merge_points(), pv.UnstructuredGrid)
assert_types(unstructured().merge_points(), pv.UnstructuredGrid)
assert_types(explicit_structured().merge_points(), pv.UnstructuredGrid)
assert_types(pointset().merge_points(), pv.PointSet)
