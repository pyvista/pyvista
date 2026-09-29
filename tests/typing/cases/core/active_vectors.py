"""Typing cases for :attr:`pyvista.DataSetAttributes.active_vectors`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import poly

assert_types(poly().point_data.active_vectors, pv.pyvista_ndarray | None)
