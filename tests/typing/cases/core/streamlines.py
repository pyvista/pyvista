"""Typing cases for :meth:`pyvista.DataSetFilters.streamlines`."""

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

assert_types(with_arrays(poly()).streamlines(vectors='v', n_points=4, source_radius=0.5), pv.PolyData)
assert_types(with_arrays(image()).streamlines(vectors='v', n_points=4, source_radius=0.5), pv.PolyData)
assert_types(with_arrays(rectilinear()).streamlines(vectors='v', n_points=4, source_radius=0.5), pv.PolyData)
assert_types(with_arrays(structured()).streamlines(vectors='v', n_points=4, source_radius=0.5), pv.PolyData)
assert_types(with_arrays(unstructured()).streamlines(vectors='v', n_points=4, source_radius=0.5), pv.PolyData)
assert_types(with_arrays(explicit_structured()).streamlines(vectors='v', n_points=4, source_radius=0.5), pv.PolyData)
assert_types(with_arrays(pointset()).streamlines(vectors='v', n_points=4, source_radius=0.5), pv.PolyData)

assert_types(with_arrays(poly()).streamlines(vectors='v', n_points=4, source_radius=0.5, return_source=True), tuple[pv.PolyData, pv.PolyData])
assert_types(with_arrays(image()).streamlines(vectors='v', n_points=4, source_radius=0.5, return_source=True), tuple[pv.PolyData, pv.PolyData])
assert_types(with_arrays(rectilinear()).streamlines(vectors='v', n_points=4, source_radius=0.5, return_source=True), tuple[pv.PolyData, pv.PolyData])
assert_types(with_arrays(structured()).streamlines(vectors='v', n_points=4, source_radius=0.5, return_source=True), tuple[pv.PolyData, pv.PolyData])
assert_types(with_arrays(unstructured()).streamlines(vectors='v', n_points=4, source_radius=0.5, return_source=True), tuple[pv.PolyData, pv.PolyData])
assert_types(with_arrays(explicit_structured()).streamlines(vectors='v', n_points=4, source_radius=0.5, return_source=True), tuple[pv.PolyData, pv.PolyData])
assert_types(with_arrays(pointset()).streamlines(vectors='v', n_points=4, source_radius=0.5, return_source=True), tuple[pv.PolyData, pv.PolyData])
