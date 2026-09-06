"""Typing cases for :func:`pyvista.plotting.utilities.algorithms.algorithm_to_mesh_handler`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from pyvista import _vtk
from pyvista.plotting.utilities.algorithms import algorithm_to_mesh_handler

_Handled = tuple[pv.DataSet, _vtk.vtkAlgorithm | _vtk.vtkAlgorithmOutput | None]

# An output port does not own its producer, so the source is held here instead.
SOURCE = _vtk.vtkSphereSource()
SOURCE.Update()

# fmt: off

assert_types(algorithm_to_mesh_handler(pv.Sphere()),           _Handled)
assert_types(algorithm_to_mesh_handler(SOURCE),                _Handled)
assert_types(algorithm_to_mesh_handler(SOURCE, 0),             _Handled)
assert_types(algorithm_to_mesh_handler(SOURCE.GetOutputPort()), _Handled)
