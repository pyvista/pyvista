"""Typing cases for :meth:`pyvista.DataSetFilters.texture_map_to_plane`."""

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

assert_types(poly().texture_map_to_plane(), pv.PolyData)
assert_types(image().texture_map_to_plane(), pv.ImageData)
assert_types(rectilinear().texture_map_to_plane(), pv.RectilinearGrid)
assert_types(structured().texture_map_to_plane(), pv.StructuredGrid)
assert_types(unstructured().texture_map_to_plane(), pv.UnstructuredGrid)
assert_types(explicit_structured().texture_map_to_plane(), pv.ExplicitStructuredGrid)
assert_types(pointset().texture_map_to_plane(), pv.PointSet)

assert_types(poly().texture_map_to_plane(inplace=True), pv.PolyData)
assert_types(image().texture_map_to_plane(inplace=True), pv.ImageData)
assert_types(rectilinear().texture_map_to_plane(inplace=True), pv.RectilinearGrid)
assert_types(structured().texture_map_to_plane(inplace=True), pv.StructuredGrid)
assert_types(unstructured().texture_map_to_plane(inplace=True), pv.UnstructuredGrid)
assert_types(explicit_structured().texture_map_to_plane(inplace=True), pv.ExplicitStructuredGrid)
assert_types(pointset().texture_map_to_plane(inplace=True), pv.PointSet)
