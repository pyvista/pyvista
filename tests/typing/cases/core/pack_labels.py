"""Typing cases for :meth:`pyvista.DataSetFilters.pack_labels`."""

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

assert_types(with_arrays(poly()).pack_labels(scalars='labels'), pv.PolyData)
assert_types(with_arrays(image()).pack_labels(scalars='labels'), pv.ImageData)
assert_types(with_arrays(rectilinear()).pack_labels(scalars='labels'), pv.RectilinearGrid)
assert_types(with_arrays(structured()).pack_labels(scalars='labels'), pv.StructuredGrid)
assert_types(with_arrays(unstructured()).pack_labels(scalars='labels'), pv.UnstructuredGrid)
assert_types(with_arrays(explicit_structured()).pack_labels(scalars='labels'), pv.ExplicitStructuredGrid)
assert_types(with_arrays(pointset()).pack_labels(scalars='labels'), pv.PointSet)

assert_types(with_arrays(poly()).pack_labels(scalars='labels', inplace=True), pv.PolyData)
assert_types(with_arrays(image()).pack_labels(scalars='labels', inplace=True), pv.ImageData)
assert_types(with_arrays(rectilinear()).pack_labels(scalars='labels', inplace=True), pv.RectilinearGrid)
assert_types(with_arrays(structured()).pack_labels(scalars='labels', inplace=True), pv.StructuredGrid)
assert_types(with_arrays(unstructured()).pack_labels(scalars='labels', inplace=True), pv.UnstructuredGrid)
assert_types(with_arrays(explicit_structured()).pack_labels(scalars='labels', inplace=True), pv.ExplicitStructuredGrid)
assert_types(with_arrays(pointset()).pack_labels(scalars='labels', inplace=True), pv.PointSet)
