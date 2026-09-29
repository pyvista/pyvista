"""Typing cases for :func:`pyvista.plotting.utilities.algorithm_to_mesh_handler`."""

from __future__ import annotations

from type_assert import assert_types

import pyvista as pv
from pyvista import _vtk
from pyvista.plotting.utilities.algorithms import algorithm_to_mesh_handler

_Algorithm = _vtk.vtkAlgorithm | _vtk.vtkAlgorithmOutput

_source = _vtk.vtkSphereSource()


def an_output_port() -> _vtk.vtkAlgorithmOutput:
    """Return an output port whose producer stays alive."""
    return _source.GetOutputPort()


# An algorithm always comes back alongside its own mesh
assert_types(algorithm_to_mesh_handler(_source), tuple[pv.DataSet, _Algorithm])
assert_types(algorithm_to_mesh_handler(an_output_port()), tuple[pv.DataSet, _Algorithm])

# A mesh is handed back unchanged, with no algorithm
assert_types(algorithm_to_mesh_handler(pv.Sphere()), tuple[pv.DataSet, _Algorithm | None])
