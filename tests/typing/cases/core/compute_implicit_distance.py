"""Typing cases for :meth:`pyvista.DataSetFilters.compute_implicit_distance`."""

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

assert_types(poly().compute_implicit_distance(pv.Sphere()), pv.PolyData)
assert_types(image().compute_implicit_distance(pv.Sphere()), pv.ImageData)
assert_types(rectilinear().compute_implicit_distance(pv.Sphere()), pv.RectilinearGrid)
assert_types(structured().compute_implicit_distance(pv.Sphere()), pv.StructuredGrid)
assert_types(unstructured().compute_implicit_distance(pv.Sphere()), pv.UnstructuredGrid)
assert_types(explicit_structured().compute_implicit_distance(pv.Sphere()), pv.ExplicitStructuredGrid)
assert_types(pointset().compute_implicit_distance(pv.Sphere()), pv.PointSet)

assert_types(poly().compute_implicit_distance(pv.Sphere(), inplace=True), pv.PolyData)
assert_types(image().compute_implicit_distance(pv.Sphere(), inplace=True), pv.ImageData)
assert_types(rectilinear().compute_implicit_distance(pv.Sphere(), inplace=True), pv.RectilinearGrid)
assert_types(structured().compute_implicit_distance(pv.Sphere(), inplace=True), pv.StructuredGrid)
assert_types(unstructured().compute_implicit_distance(pv.Sphere(), inplace=True), pv.UnstructuredGrid)
assert_types(explicit_structured().compute_implicit_distance(pv.Sphere(), inplace=True), pv.ExplicitStructuredGrid)
assert_types(pointset().compute_implicit_distance(pv.Sphere(), inplace=True), pv.PointSet)
