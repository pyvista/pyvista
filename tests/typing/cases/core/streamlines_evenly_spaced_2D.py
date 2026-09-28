"""Typing cases for :meth:`pyvista.DataSetFilters.streamlines_evenly_spaced_2D`."""

from __future__ import annotations

import numpy as np
from type_assert import assert_types
from typing_extensions import Never

import pyvista as pv
from tests.typing.meshes import explicit_structured
from tests.typing.meshes import pointset
from tests.typing.meshes import with_arrays

SKIP_RUNTIME = {
    "with_arrays(explicit_structured()).streamlines_evenly_spaced_2D(vectors='v', start_position=(0.5, 0.5, 0.0))": 'an `ExplicitStructuredGrid` is 3D, so it cannot lie in the XY plane this filter needs',
    'pointset().streamlines_evenly_spaced_2D()': 'a `PointSet` has no cells, so the call raises',
}

AXIS = np.arange(4, dtype=float)


def a_plane() -> pv.PolyData:
    """Return a plane in the XY plane."""
    return pv.Plane(center=(1.5, 1.5, 0.0), i_size=3, j_size=3, i_resolution=3, j_resolution=3)


def a_slice() -> pv.ImageData:
    """Return a single-slice uniform grid."""
    return pv.ImageData(dimensions=(4, 4, 1))


def a_rectilinear_slice() -> pv.RectilinearGrid:
    """Return a single-slice rectilinear grid."""
    return pv.RectilinearGrid(AXIS, AXIS, np.array([0.0]))


def a_structured_slice() -> pv.StructuredGrid:
    """Return a single-slice structured grid."""
    x, y, z = np.meshgrid(AXIS, AXIS, np.array([0.0]), indexing='ij')
    return pv.StructuredGrid(x, y, z)


def an_unstructured_slice() -> pv.UnstructuredGrid:
    """Return a single-slice unstructured grid."""
    return a_slice().cast_to_unstructured_grid()


assert_types(with_arrays(a_plane()).streamlines_evenly_spaced_2D(vectors='v', start_position=(0.5, 0.5, 0.0)), pv.PolyData)
assert_types(with_arrays(a_slice()).streamlines_evenly_spaced_2D(vectors='v', start_position=(0.5, 0.5, 0.0)), pv.PolyData)
assert_types(with_arrays(a_rectilinear_slice()).streamlines_evenly_spaced_2D(vectors='v', start_position=(0.5, 0.5, 0.0)), pv.PolyData)
assert_types(with_arrays(a_structured_slice()).streamlines_evenly_spaced_2D(vectors='v', start_position=(0.5, 0.5, 0.0)), pv.PolyData)
assert_types(with_arrays(an_unstructured_slice()).streamlines_evenly_spaced_2D(vectors='v', start_position=(0.5, 0.5, 0.0)), pv.PolyData)
assert_types(with_arrays(explicit_structured()).streamlines_evenly_spaced_2D(vectors='v', start_position=(0.5, 0.5, 0.0)), pv.PolyData)
assert_types(pointset().streamlines_evenly_spaced_2D(), Never)  # pragma: no cover
