"""Typing cases for :meth:`pyvista.StructuredGridFilters.extract_subset`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import structured

# A subset of a structured grid is still structured
assert_types(structured().extract_subset((0, 2, 0, 2, 0, 2)), pv.StructuredGrid)
assert_types(structured().extract_subset((0, 3, 0, 3, 0, 3), rate=(2, 2, 2)), pv.StructuredGrid)
assert_types(
    structured().extract_subset((0, 3, 0, 3, 0, 3), rate=(2, 2, 2), boundary=True),
    pv.StructuredGrid,
)
