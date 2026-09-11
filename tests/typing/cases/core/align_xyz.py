"""Typing cases for :meth:`pyvista.DataSetFilters.align_xyz`."""

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

assert_types(poly().align_xyz(), pv.PolyData)
assert_types(image().align_xyz(), pv.ImageData)
assert_types(rectilinear().align_xyz(), pv.RectilinearGrid)
assert_types(structured().align_xyz(), pv.StructuredGrid)
assert_types(unstructured().align_xyz(), pv.UnstructuredGrid)
assert_types(explicit_structured().align_xyz(), pv.ExplicitStructuredGrid)
assert_types(pointset().align_xyz(), pv.PointSet)

assert_types(poly().align_xyz(return_matrix=True), tuple[pv.PolyData, NumpyArray[float]])
assert_types(image().align_xyz(return_matrix=True), tuple[pv.ImageData, NumpyArray[float]])
assert_types(rectilinear().align_xyz(return_matrix=True), tuple[pv.RectilinearGrid, NumpyArray[float]])
assert_types(structured().align_xyz(return_matrix=True), tuple[pv.StructuredGrid, NumpyArray[float]])
assert_types(unstructured().align_xyz(return_matrix=True), tuple[pv.UnstructuredGrid, NumpyArray[float]])
assert_types(explicit_structured().align_xyz(return_matrix=True), tuple[pv.ExplicitStructuredGrid, NumpyArray[float]])
assert_types(pointset().align_xyz(return_matrix=True), tuple[pv.PointSet, NumpyArray[float]])
