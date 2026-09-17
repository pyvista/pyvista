"""Typing cases for :meth:`pyvista.PolyDataFilters.merge`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import poly
from tests.typing.meshes import unstructured

# Merging polydata stays polydata, and so does an in-place merge
assert_types(poly().merge(poly()), pv.PolyData)
assert_types(poly().merge([poly(), poly()]), pv.PolyData)
assert_types(poly().merge(poly(), inplace=True), pv.PolyData)
assert_types(poly().merge(unstructured()), pv.UnstructuredGrid)
