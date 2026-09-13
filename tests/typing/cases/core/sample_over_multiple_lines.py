"""Typing cases for :meth:`pyvista.DataSetFilters.sample_over_multiple_lines`."""

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

assert_types(poly().sample_over_multiple_lines([[-1, 0, 0], [0, 0, 0], [1, 0, 0]]), pv.PolyData)
assert_types(image().sample_over_multiple_lines([[-1, 0, 0], [0, 0, 0], [1, 0, 0]]), pv.PolyData)
assert_types(rectilinear().sample_over_multiple_lines([[-1, 0, 0], [0, 0, 0], [1, 0, 0]]), pv.PolyData)
assert_types(structured().sample_over_multiple_lines([[-1, 0, 0], [0, 0, 0], [1, 0, 0]]), pv.PolyData)
assert_types(unstructured().sample_over_multiple_lines([[-1, 0, 0], [0, 0, 0], [1, 0, 0]]), pv.PolyData)
assert_types(explicit_structured().sample_over_multiple_lines([[-1, 0, 0], [0, 0, 0], [1, 0, 0]]), pv.PolyData)
assert_types(pointset().sample_over_multiple_lines([[-1, 0, 0], [0, 0, 0], [1, 0, 0]]), pv.PolyData)
