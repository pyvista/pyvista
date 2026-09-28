"""Typing cases for :meth:`pyvista.DataSetFilters.extract_points`."""

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

assert_types(poly().extract_points([0, 1, 2]), pv.UnstructuredGrid)
assert_types(image().extract_points([0, 1, 2]), pv.UnstructuredGrid)
assert_types(rectilinear().extract_points([0, 1, 2]), pv.UnstructuredGrid)
assert_types(structured().extract_points([0, 1, 2]), pv.UnstructuredGrid)
assert_types(unstructured().extract_points([0, 1, 2]), pv.UnstructuredGrid)
assert_types(explicit_structured().extract_points([0, 1, 2]), pv.UnstructuredGrid)
assert_types(pointset().extract_points([0, 1, 2]), pv.PointSet)
