"""Typing cases for :meth:`pyvista.UnstructuredGridFilters.clean`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import unstructured

# Cleaning keeps the grid a grid, whichever knobs are set
assert_types(unstructured().clean(), pv.UnstructuredGrid)
assert_types(unstructured().clean(tolerance=1e-6), pv.UnstructuredGrid)
assert_types(unstructured().clean(remove_unused_points=False), pv.UnstructuredGrid)
assert_types(unstructured().clean(produce_merge_map=False), pv.UnstructuredGrid)
assert_types(unstructured().clean(average_point_data=False), pv.UnstructuredGrid)
