"""Typing cases for :meth:`pyvista.DataObjectFilters.cell_quality`."""

from __future__ import annotations

from type_assert import assert_types
from typing_extensions import Never

import pyvista as pv
from tests.typing.meshes import pointset

SKIP_RUNTIME = {
    'pointset().cell_quality()': 'a `PointSet` has no cells, so the call raises',
}


def a_grid() -> pv.ImageData:
    """Return a small grid."""
    return pv.ImageData(dimensions=(3, 3, 3))


def a_multiblock() -> pv.MultiBlock:
    """Return a `MultiBlock` holding one mesh."""
    return pv.MultiBlock([pv.Sphere()])


assert_types(pv.Sphere().cell_quality(), pv.PolyData)
assert_types(a_grid().cell_quality(), pv.ImageData)
assert_types(a_multiblock().cell_quality(), pv.MultiBlock)

assert_types(pointset().cell_quality(), Never)  # pragma: no cover
