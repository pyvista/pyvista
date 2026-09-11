"""Typing cases for :meth:`pyvista.DataSetFilters.remove_points`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import explicit_structured
from tests.typing.meshes import image
from tests.typing.meshes import pointset
from tests.typing.meshes import rectilinear
from tests.typing.meshes import structured
from tests.typing.meshes import unstructured

assert_types(image().remove_points([0]), pv.UnstructuredGrid)
assert_types(rectilinear().remove_points([0]), pv.UnstructuredGrid)
assert_types(structured().remove_points([0]), pv.UnstructuredGrid)
assert_types(unstructured().remove_points([0]), pv.UnstructuredGrid)
assert_types(explicit_structured().remove_points([0]), pv.UnstructuredGrid)
assert_types(pointset().remove_points([0]), pv.PointSet)
