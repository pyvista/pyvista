"""Typing cases for :meth:`pyvista.UnstructuredGridFilters.reconstruct_surface`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv


def a_grid() -> pv.UnstructuredGrid:
    """Return a coarse sphere as an unstructured grid."""
    return pv.Sphere(theta_resolution=8, phi_resolution=8).cast_to_unstructured_grid()


assert_types(a_grid().reconstruct_surface(), pv.PolyData)
