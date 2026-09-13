"""Typing cases for :meth:`pyvista.DataSetFilters.align`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from pyvista.core._typing_core import NumpyArray
from tests.typing.meshes import explicit_structured
from tests.typing.meshes import image
from tests.typing.meshes import pointset
from tests.typing.meshes import poly
from tests.typing.meshes import rectilinear
from tests.typing.meshes import structured
from tests.typing.meshes import unstructured

assert_types(poly().align(pv.Sphere()), pv.PolyData)
assert_types(image().align(pv.Sphere()), pv.ImageData)
assert_types(rectilinear().align(pv.Sphere()), pv.RectilinearGrid)
assert_types(structured().align(pv.Sphere()), pv.StructuredGrid)
assert_types(unstructured().align(pv.Sphere()), pv.UnstructuredGrid)
assert_types(explicit_structured().align(pv.Sphere()), pv.ExplicitStructuredGrid)
assert_types(pointset().align(pv.Sphere()), pv.PointSet)

assert_types(poly().align(pv.Sphere(), return_matrix=True), tuple[pv.PolyData, NumpyArray[float]])
assert_types(image().align(pv.Sphere(), return_matrix=True), tuple[pv.ImageData, NumpyArray[float]])
assert_types(rectilinear().align(pv.Sphere(), return_matrix=True), tuple[pv.RectilinearGrid, NumpyArray[float]])
assert_types(structured().align(pv.Sphere(), return_matrix=True), tuple[pv.StructuredGrid, NumpyArray[float]])
assert_types(unstructured().align(pv.Sphere(), return_matrix=True), tuple[pv.UnstructuredGrid, NumpyArray[float]])
assert_types(explicit_structured().align(pv.Sphere(), return_matrix=True), tuple[pv.ExplicitStructuredGrid, NumpyArray[float]])
assert_types(pointset().align(pv.Sphere(), return_matrix=True), tuple[pv.PointSet, NumpyArray[float]])
