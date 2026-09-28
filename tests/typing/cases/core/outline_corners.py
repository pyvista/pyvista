"""Typing cases for :meth:`pyvista.DataSetFilters.outline_corners`."""

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

assert_types(poly().outline_corners(), pv.PolyData)
assert_types(image().outline_corners(), pv.PolyData)
assert_types(rectilinear().outline_corners(), pv.PolyData)
assert_types(structured().outline_corners(), pv.PolyData)
assert_types(unstructured().outline_corners(), pv.PolyData)
assert_types(explicit_structured().outline_corners(), pv.PolyData)
assert_types(pointset().outline_corners(), pv.PolyData)
