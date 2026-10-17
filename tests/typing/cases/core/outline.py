"""Typing cases for :meth:`pyvista.DataSetFilters.outline`."""

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

assert_types(poly().outline(), pv.PolyData)
assert_types(image().outline(), pv.PolyData)
assert_types(rectilinear().outline(), pv.PolyData)
assert_types(structured().outline(), pv.PolyData)
assert_types(unstructured().outline(), pv.PolyData)
assert_types(explicit_structured().outline(), pv.PolyData)
assert_types(pointset().outline(), pv.PolyData)

assert_types(poly().outline(generate_faces=True), pv.PolyData)
assert_types(image().outline(generate_faces=True), pv.PolyData)
assert_types(rectilinear().outline(generate_faces=True), pv.PolyData)
assert_types(structured().outline(generate_faces=True), pv.PolyData)
assert_types(unstructured().outline(generate_faces=True), pv.PolyData)
assert_types(explicit_structured().outline(generate_faces=True), pv.PolyData)
assert_types(pointset().outline(generate_faces=True), pv.PolyData)
