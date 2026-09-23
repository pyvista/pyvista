"""Typing cases for :meth:`pyvista.DataSetFilters.integrate_data`."""

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

assert_types(poly().integrate_data(), pv.UnstructuredGrid)
assert_types(image().integrate_data(), pv.UnstructuredGrid)
assert_types(rectilinear().integrate_data(), pv.UnstructuredGrid)
assert_types(structured().integrate_data(), pv.UnstructuredGrid)
assert_types(unstructured().integrate_data(), pv.UnstructuredGrid)
assert_types(explicit_structured().integrate_data(), pv.UnstructuredGrid)
assert_types(pointset().integrate_data(), pv.UnstructuredGrid)
