"""Typing cases for :meth:`pyvista.PolyDataFilters.__add__`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import multiblock
from tests.typing.meshes import poly
from tests.typing.meshes import unstructured

# Adding polydata gives back polydata, and anything else gives what it merges into
assert_types(poly() + poly(), pv.PolyData)
assert_types(poly() + [poly(), poly()], pv.PolyData)  # noqa: RUF005
assert_types(poly() + unstructured(), pv.UnstructuredGrid)
assert_types(poly() + multiblock(), pv.PolyData | pv.UnstructuredGrid)
