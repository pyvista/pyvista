"""Typing cases for :attr:`pyvista.DataSet.active_tensors`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import poly

assert_types(poly().active_tensors, pv.pyvista_ndarray | None)
