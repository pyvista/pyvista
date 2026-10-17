"""Typing cases for :meth:`pyvista.DataSetFilters.delaunay_3d`."""

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

assert_types(poly().delaunay_3d(), pv.UnstructuredGrid)
assert_types(image().delaunay_3d(), pv.UnstructuredGrid)
assert_types(rectilinear().delaunay_3d(), pv.UnstructuredGrid)
assert_types(structured().delaunay_3d(), pv.UnstructuredGrid)
assert_types(unstructured().delaunay_3d(), pv.UnstructuredGrid)
assert_types(explicit_structured().delaunay_3d(), pv.UnstructuredGrid)
assert_types(pointset().delaunay_3d(), pv.UnstructuredGrid)
