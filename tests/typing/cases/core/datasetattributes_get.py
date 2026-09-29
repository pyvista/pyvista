"""Typing cases for :meth:`pyvista.DataSetAttributes.get`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import poly

assert_types(poly().point_data.get('Normals'), pv.pyvista_ndarray | None)
assert_types(poly().point_data.get('missing'), pv.pyvista_ndarray | None)
assert_types(poly().point_data.get('missing', 0), pv.pyvista_ndarray | int)
assert_types(poly().point_data.get('Normals', 'default'), pv.pyvista_ndarray | str)
