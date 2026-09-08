"""Typing cases for :meth:`pyvista.DataObjectFilters.validate_mesh`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from pyvista.core.filters.data_object import _MeshValidationReport


def a_multiblock() -> pv.MultiBlock:
    """Return a `MultiBlock` holding one mesh."""
    return pv.MultiBlock([pv.Sphere()])


assert_types(pv.Sphere().validate_mesh(), _MeshValidationReport[pv.PolyData])
assert_types(a_multiblock().validate_mesh(), _MeshValidationReport[pv.MultiBlock])
