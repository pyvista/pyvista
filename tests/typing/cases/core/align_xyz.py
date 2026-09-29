"""Typing cases for :meth:`pyvista.DataSetFilters.align_xyz`."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from type_assert import assert_types

import pyvista as pv
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

assert_types(poly().align_xyz(return_matrix=True), tuple[pv.PolyData, NDArray[np.float64]])
assert_types(image().align_xyz(return_matrix=True), tuple[pv.ImageData, NDArray[np.float64]])
assert_types(rectilinear().align_xyz(return_matrix=True), tuple[pv.RectilinearGrid, NDArray[np.float64]])
assert_types(structured().align_xyz(return_matrix=True), tuple[pv.StructuredGrid, NDArray[np.float64]])
assert_types(unstructured().align_xyz(return_matrix=True), tuple[pv.UnstructuredGrid, NDArray[np.float64]])
assert_types(explicit_structured().align_xyz(return_matrix=True), tuple[pv.ExplicitStructuredGrid, NDArray[np.float64]])
assert_types(pointset().align_xyz(return_matrix=True), tuple[pv.PointSet, NDArray[np.float64]])
