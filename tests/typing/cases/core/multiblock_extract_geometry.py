"""Typing cases for :meth:`pyvista.CompositeFilters.extract_geometry`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from tests.typing.meshes import multiblock

SKIP_RUNTIME = {
    'multiblock().extract_geometry()': 'the filter is deprecated and warns when called',
}

assert_types(multiblock().extract_geometry(), pv.PolyData)
