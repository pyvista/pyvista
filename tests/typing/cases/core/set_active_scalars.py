"""Typing cases for :meth:`pyvista.DataSet.set_active_scalars`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import poly

assert_types(poly().elevation().set_active_scalars('Elevation'), tuple[pv.FieldAssociation, pv.pyvista_ndarray | None])
assert_types(poly().set_active_scalars(None), tuple[pv.FieldAssociation, pv.pyvista_ndarray | None])
