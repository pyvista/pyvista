"""Typing cases for :meth:`pyvista.MultiBlock.set_active_scalars`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import poly

assert_types(pv.MultiBlock([poly().elevation()]).set_active_scalars('Elevation'), tuple[pv.FieldAssociation, pv.pyvista_ndarray])
