"""Typing cases for :meth:`pyvista.DataSetFilters.align`."""

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

assert_types(poly().align(pv.Sphere()), pv.PolyData)
assert_types(image().align(pv.Sphere()), pv.ImageData)
assert_types(rectilinear().align(pv.Sphere()), pv.RectilinearGrid)
assert_types(structured().align(pv.Sphere()), pv.StructuredGrid)
assert_types(unstructured().align(pv.Sphere()), pv.UnstructuredGrid)
assert_types(explicit_structured().align(pv.Sphere()), pv.ExplicitStructuredGrid)
assert_types(pointset().align(pv.Sphere()), pv.PointSet)

assert_types(poly().align(pv.Sphere(), return_matrix=True), tuple[pv.PolyData, npt.NDArray[np.float64]])
assert_types(image().align(pv.Sphere(), return_matrix=True), tuple[pv.ImageData, npt.NDArray[np.float64]])
assert_types(rectilinear().align(pv.Sphere(), return_matrix=True), tuple[pv.RectilinearGrid, npt.NDArray[np.float64]])
assert_types(structured().align(pv.Sphere(), return_matrix=True), tuple[pv.StructuredGrid, npt.NDArray[np.float64]])
assert_types(unstructured().align(pv.Sphere(), return_matrix=True), tuple[pv.UnstructuredGrid, npt.NDArray[np.float64]])
assert_types(explicit_structured().align(pv.Sphere(), return_matrix=True), tuple[pv.ExplicitStructuredGrid, npt.NDArray[np.float64]])
assert_types(pointset().align(pv.Sphere(), return_matrix=True), tuple[pv.PointSet, npt.NDArray[np.float64]])
