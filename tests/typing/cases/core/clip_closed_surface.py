"""Typing cases for :meth:`pyvista.PolyDataFilters.clip_closed_surface`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv


def poly() -> pv.PolyData:
    """Return a sphere."""
    return pv.Sphere(theta_resolution=8, phi_resolution=8)


assert_types(poly().clip_closed_surface(), pv.PolyData)
