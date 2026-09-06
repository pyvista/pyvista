"""Typing cases for :func:`pyvista.plotting._plotting._resolve_scalars_field`."""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv
from pyvista.core.utilities.arrays import CellLiteral
from pyvista.core.utilities.arrays import FieldAssociation
from pyvista.core.utilities.arrays import PointLiteral
from pyvista.plotting._plotting import _resolve_scalars_field


def a_mesh() -> pv.PolyData:
    """Return a mesh whose point and cell counts differ."""
    return pv.Sphere()


# fmt: off

assert_types(_resolve_scalars_field(np.zeros(a_mesh().n_points), a_mesh(), 'point'),  PointLiteral | CellLiteral)
assert_types(_resolve_scalars_field(np.zeros(a_mesh().n_cells), a_mesh(), 'cell'),    PointLiteral | CellLiteral)
assert_types(_resolve_scalars_field(np.zeros(a_mesh().n_points), a_mesh(), FieldAssociation.POINT), PointLiteral | CellLiteral)
assert_types(_resolve_scalars_field(np.zeros(a_mesh().n_cells), a_mesh(), FieldAssociation.CELL),   PointLiteral | CellLiteral)
