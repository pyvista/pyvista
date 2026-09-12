"""Typing cases for :func:`pyvista.generate_plane`."""

from __future__ import annotations

import numpy as np
from type_assert import assert_types

import pyvista as pv
from pyvista import _vtk

# A raw VTK object, not a PyVista wrapper
assert_types(pv.generate_plane((0.0, 0.0, 1.0), (0.0, 0.0, 0.0)), _vtk.vtkPlane)
assert_types(pv.generate_plane(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])), _vtk.vtkPlane)
