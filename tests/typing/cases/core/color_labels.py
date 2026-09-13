"""Typing cases for :meth:`pyvista.DataSetFilters.color_labels`."""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv
from pyvista.plotting._typing import ColorLike
from tests.typing.meshes import explicit_structured
from tests.typing.meshes import image
from tests.typing.meshes import pointset
from tests.typing.meshes import poly
from tests.typing.meshes import rectilinear
from tests.typing.meshes import structured
from tests.typing.meshes import unstructured
from tests.typing.meshes import with_arrays

assert_types(with_arrays(poly()).color_labels(scalars='labels'), pv.PolyData)
assert_types(with_arrays(image()).color_labels(scalars='labels'), pv.ImageData)
assert_types(with_arrays(rectilinear()).color_labels(scalars='labels'), pv.RectilinearGrid)
assert_types(with_arrays(structured()).color_labels(scalars='labels'), pv.StructuredGrid)
assert_types(with_arrays(unstructured()).color_labels(scalars='labels'), pv.UnstructuredGrid)
assert_types(with_arrays(explicit_structured()).color_labels(scalars='labels'), pv.ExplicitStructuredGrid)
assert_types(with_arrays(pointset()).color_labels(scalars='labels'), pv.PointSet)

assert_types(with_arrays(poly()).color_labels(scalars='labels', return_dict=True), tuple[pv.PolyData, dict[float | np.integer | np.floating, ColorLike]])
assert_types(with_arrays(image()).color_labels(scalars='labels', return_dict=True), tuple[pv.ImageData, dict[float | np.integer | np.floating, ColorLike]])
assert_types(with_arrays(rectilinear()).color_labels(scalars='labels', return_dict=True), tuple[pv.RectilinearGrid, dict[float | np.integer | np.floating, ColorLike]])
assert_types(with_arrays(structured()).color_labels(scalars='labels', return_dict=True), tuple[pv.StructuredGrid, dict[float | np.integer | np.floating, ColorLike]])
assert_types(with_arrays(unstructured()).color_labels(scalars='labels', return_dict=True), tuple[pv.UnstructuredGrid, dict[float | np.integer | np.floating, ColorLike]])
assert_types(with_arrays(explicit_structured()).color_labels(scalars='labels', return_dict=True), tuple[pv.ExplicitStructuredGrid, dict[float | np.integer | np.floating, ColorLike]])
assert_types(with_arrays(pointset()).color_labels(scalars='labels', return_dict=True), tuple[pv.PointSet, dict[float | np.integer | np.floating, ColorLike]])
