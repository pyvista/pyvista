"""Typing cases for :meth:`pyvista.DataSetFilters.align_xyz`."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
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

assert_types(poly().align_xyz(return_matrix=True), tuple[pv.PolyData, npt.NDArray[np.float64]])
assert_types(image().align_xyz(return_matrix=True), tuple[pv.ImageData, npt.NDArray[np.float64]])
assert_types(rectilinear().align_xyz(return_matrix=True), tuple[pv.RectilinearGrid, npt.NDArray[np.float64]])
assert_types(structured().align_xyz(return_matrix=True), tuple[pv.StructuredGrid, npt.NDArray[np.float64]])
assert_types(unstructured().align_xyz(return_matrix=True), tuple[pv.UnstructuredGrid, npt.NDArray[np.float64]])
assert_types(explicit_structured().align_xyz(return_matrix=True), tuple[pv.ExplicitStructuredGrid, npt.NDArray[np.float64]])
assert_types(pointset().align_xyz(return_matrix=True), tuple[pv.PointSet, npt.NDArray[np.float64]])
