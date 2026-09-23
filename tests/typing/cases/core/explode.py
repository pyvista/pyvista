"""Typing cases for :meth:`pyvista.DataSetFilters.explode`."""

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

assert_types(poly().explode(), pv.UnstructuredGrid)
assert_types(image().explode(), pv.UnstructuredGrid)
assert_types(rectilinear().explode(), pv.UnstructuredGrid)
assert_types(structured().explode(), pv.UnstructuredGrid)
assert_types(unstructured().explode(), pv.UnstructuredGrid)
assert_types(explicit_structured().explode(), pv.UnstructuredGrid)
assert_types(pointset().explode(), pv.PointSet)
