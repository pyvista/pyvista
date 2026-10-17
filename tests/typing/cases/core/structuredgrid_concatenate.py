"""Typing cases for :meth:`pyvista.StructuredGrid.concatenate`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import structured


def shifted() -> pv.StructuredGrid:
    """Return a structured grid placed beside the shared one."""
    return structured().translate((3.0, 0.0, 0.0))


# Two structured grids join into one
assert_types(structured().concatenate(shifted(), axis=0), pv.StructuredGrid)
assert_types(structured().concatenate(shifted(), axis=0, tolerance=1e-6), pv.StructuredGrid)
