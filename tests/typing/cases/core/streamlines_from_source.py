"""Typing cases for :meth:`pyvista.DataSetFilters.streamlines_from_source`."""

from __future__ import annotations

from typing import Never

from type_assert import assert_types

import pyvista as pv
from pyvista import _vtk
from tests.typing.meshes import explicit_structured
from tests.typing.meshes import image
from tests.typing.meshes import pointset
from tests.typing.meshes import poly
from tests.typing.meshes import rectilinear
from tests.typing.meshes import structured
from tests.typing.meshes import unstructured
from tests.typing.meshes import with_arrays


def raw_source() -> _vtk.vtkPolyData:
    """Return a single seed point as a bare VTK polydata."""
    raw = _vtk.vtkPolyData()
    raw.ShallowCopy(pv.PolyData([[0.1, 0.1, 0.1]]))
    return raw


assert_types(with_arrays(poly()).streamlines_from_source(pv.PolyData([[0.1, 0.1, 0.1]]), vectors='v'), pv.PolyData)
assert_types(with_arrays(image()).streamlines_from_source(pv.PolyData([[0.1, 0.1, 0.1]]), vectors='v'), pv.PolyData)
assert_types(with_arrays(rectilinear()).streamlines_from_source(pv.PolyData([[0.1, 0.1, 0.1]]), vectors='v'), pv.PolyData)
assert_types(with_arrays(structured()).streamlines_from_source(pv.PolyData([[0.1, 0.1, 0.1]]), vectors='v'), pv.PolyData)
assert_types(with_arrays(unstructured()).streamlines_from_source(pv.PolyData([[0.1, 0.1, 0.1]]), vectors='v'), pv.PolyData)
assert_types(with_arrays(explicit_structured()).streamlines_from_source(pv.PolyData([[0.1, 0.1, 0.1]]), vectors='v'), pv.PolyData)

# A bare VTK source is wrapped
assert_types(with_arrays(poly()).streamlines_from_source(raw_source(), vectors='v'), pv.PolyData)

assert_types(pointset().streamlines_from_source(), Never)
