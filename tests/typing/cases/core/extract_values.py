"""Typing cases for :meth:`pyvista.DataSetFilters.extract_values`."""

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
from tests.typing.meshes import with_arrays

assert_types(with_arrays(poly()).extract_values([0, 1], scalars='labels'), pv.UnstructuredGrid)
assert_types(with_arrays(image()).extract_values([0, 1], scalars='labels'), pv.UnstructuredGrid)
assert_types(with_arrays(rectilinear()).extract_values([0, 1], scalars='labels'), pv.UnstructuredGrid)
assert_types(with_arrays(structured()).extract_values([0, 1], scalars='labels'), pv.UnstructuredGrid)
assert_types(with_arrays(unstructured()).extract_values([0, 1], scalars='labels'), pv.UnstructuredGrid)
assert_types(with_arrays(explicit_structured()).extract_values([0, 1], scalars='labels'), pv.UnstructuredGrid)
assert_types(with_arrays(pointset()).extract_values([0, 1], scalars='labels'), pv.PointSet)

assert_types(with_arrays(poly()).extract_values([0, 1], scalars='labels', split=True), pv.MultiBlock)
assert_types(with_arrays(image()).extract_values([0, 1], scalars='labels', split=True), pv.MultiBlock)
assert_types(with_arrays(rectilinear()).extract_values([0, 1], scalars='labels', split=True), pv.MultiBlock)
assert_types(with_arrays(structured()).extract_values([0, 1], scalars='labels', split=True), pv.MultiBlock)
assert_types(with_arrays(unstructured()).extract_values([0, 1], scalars='labels', split=True), pv.MultiBlock)
assert_types(with_arrays(explicit_structured()).extract_values([0, 1], scalars='labels', split=True), pv.MultiBlock)
assert_types(with_arrays(pointset()).extract_values([0, 1], scalars='labels', split=True), pv.MultiBlock)
