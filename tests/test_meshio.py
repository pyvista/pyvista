import pytest

import os

import numpy

import pyvista

from pyvista import examples


def test_beam():
    # Load reference mesh
    mesh_ref = pyvista.UnstructuredGrid(examples.hexbeamfile)

    # Save and read reference mesh using meshio
    pyvista.save_meshio("test_meshio.vtk", mesh_ref)
    mesh = pyvista.read_meshio("test_meshio.vtk")
    os.remove("test_meshio.vtk")

    # Assert mesh is still the same
    assert numpy.allclose(mesh_ref.points, mesh.points)
    assert numpy.allclose(mesh_ref.cells, mesh.cells)
    for k, v in mesh_ref.point_arrays.items():
        assert numpy.allclose(v, mesh.point_arrays[k])
    for k, v in mesh_ref.cell_arrays.items():
        assert numpy.allclose(v, mesh.cell_arrays[k])