"""Typing cases for :meth:`pyvista.DataSetFilters.streamlines_from_source`."""

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

assert_types(with_arrays(poly()).streamlines_from_source(pv.PolyData([[0.1, 0.1, 0.1]]), vectors='v'), pv.PolyData)
assert_types(with_arrays(image()).streamlines_from_source(pv.PolyData([[0.1, 0.1, 0.1]]), vectors='v'), pv.PolyData)
assert_types(with_arrays(rectilinear()).streamlines_from_source(pv.PolyData([[0.1, 0.1, 0.1]]), vectors='v'), pv.PolyData)
assert_types(with_arrays(structured()).streamlines_from_source(pv.PolyData([[0.1, 0.1, 0.1]]), vectors='v'), pv.PolyData)
assert_types(with_arrays(unstructured()).streamlines_from_source(pv.PolyData([[0.1, 0.1, 0.1]]), vectors='v'), pv.PolyData)
assert_types(with_arrays(explicit_structured()).streamlines_from_source(pv.PolyData([[0.1, 0.1, 0.1]]), vectors='v'), pv.PolyData)
assert_types(with_arrays(pointset()).streamlines_from_source(pv.PolyData([[0.1, 0.1, 0.1]]), vectors='v'), pv.PolyData)
