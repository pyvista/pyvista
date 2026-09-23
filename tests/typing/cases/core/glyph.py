"""Typing cases for :meth:`pyvista.DataSetFilters.glyph`."""

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
from tests.typing.meshes import with_arrays

assert_types(with_arrays(poly()).glyph(orient='v', scale=False), pv.PolyData)
assert_types(with_arrays(image()).glyph(orient='v', scale=False), pv.PolyData)
assert_types(with_arrays(rectilinear()).glyph(orient='v', scale=False), pv.PolyData)
assert_types(with_arrays(structured()).glyph(orient='v', scale=False), pv.PolyData)
assert_types(with_arrays(unstructured()).glyph(orient='v', scale=False), pv.PolyData)
assert_types(with_arrays(explicit_structured()).glyph(orient='v', scale=False), pv.PolyData)
assert_types(with_arrays(pointset()).glyph(orient='v', scale=False), pv.PolyData)
