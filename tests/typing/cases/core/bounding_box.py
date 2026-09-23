"""Typing cases for :meth:`pyvista.DataSetFilters.bounding_box`."""

from __future__ import annotations

import numpy as np
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

assert_types(poly().bounding_box(), pv.MultiBlock)
assert_types(image().bounding_box(), pv.MultiBlock)
assert_types(rectilinear().bounding_box(), pv.MultiBlock)
assert_types(structured().bounding_box(), pv.MultiBlock)
assert_types(unstructured().bounding_box(), pv.MultiBlock)
assert_types(explicit_structured().bounding_box(), pv.MultiBlock)
assert_types(pointset().bounding_box(), pv.MultiBlock)

assert_types(poly().bounding_box(as_composite=False), pv.PolyData)
assert_types(image().bounding_box(as_composite=False), pv.PolyData)
assert_types(rectilinear().bounding_box(as_composite=False), pv.PolyData)
assert_types(structured().bounding_box(as_composite=False), pv.PolyData)
assert_types(unstructured().bounding_box(as_composite=False), pv.PolyData)
assert_types(explicit_structured().bounding_box(as_composite=False), pv.PolyData)
assert_types(pointset().bounding_box(as_composite=False), pv.PolyData)

assert_types(poly().bounding_box(return_meta=True), tuple[pv.MultiBlock, NumpyArray[np.floating], NumpyArray[np.floating]])
assert_types(image().bounding_box(return_meta=True), tuple[pv.MultiBlock, NumpyArray[np.floating], NumpyArray[np.floating]])
assert_types(rectilinear().bounding_box(return_meta=True), tuple[pv.MultiBlock, NumpyArray[np.floating], NumpyArray[np.floating]])
assert_types(structured().bounding_box(return_meta=True), tuple[pv.MultiBlock, NumpyArray[np.floating], NumpyArray[np.floating]])
assert_types(unstructured().bounding_box(return_meta=True), tuple[pv.MultiBlock, NumpyArray[np.floating], NumpyArray[np.floating]])
assert_types(explicit_structured().bounding_box(return_meta=True), tuple[pv.MultiBlock, NumpyArray[np.floating], NumpyArray[np.floating]])
assert_types(pointset().bounding_box(return_meta=True), tuple[pv.MultiBlock, NumpyArray[np.floating], NumpyArray[np.floating]])

assert_types(poly().bounding_box(as_composite=False, return_meta=True), tuple[pv.PolyData, NumpyArray[np.floating], NumpyArray[np.floating]])
assert_types(image().bounding_box(as_composite=False, return_meta=True), tuple[pv.PolyData, NumpyArray[np.floating], NumpyArray[np.floating]])
assert_types(rectilinear().bounding_box(as_composite=False, return_meta=True), tuple[pv.PolyData, NumpyArray[np.floating], NumpyArray[np.floating]])
assert_types(structured().bounding_box(as_composite=False, return_meta=True), tuple[pv.PolyData, NumpyArray[np.floating], NumpyArray[np.floating]])
assert_types(unstructured().bounding_box(as_composite=False, return_meta=True), tuple[pv.PolyData, NumpyArray[np.floating], NumpyArray[np.floating]])
assert_types(explicit_structured().bounding_box(as_composite=False, return_meta=True), tuple[pv.PolyData, NumpyArray[np.floating], NumpyArray[np.floating]])
assert_types(pointset().bounding_box(as_composite=False, return_meta=True), tuple[pv.PolyData, NumpyArray[np.floating], NumpyArray[np.floating]])
