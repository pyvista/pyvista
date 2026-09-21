"""Typing cases for :meth:`pyvista.DataSetFilters.__iadd__`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import poly
from tests.typing.meshes import unstructured

# An in-place merge keeps this mesh, which is why it accepts an unstructured grid
mesh = unstructured()
mesh += poly()
assert_types(mesh, pv.UnstructuredGrid)
