import pytest

import numpy

import pyvista

from pyvista import examples


beam = pyvista.UnstructuredGrid(examples.hexbeamfile)
airplane = examples.load_airplane().cast_to_unstructured_grid()
@pytest.mark.parametrize("mesh_in", [beam, airplane])
def test_meshio(mesh_in, tmpdir):
    # Save and read reference mesh using meshio
    filename = tmpdir.mkdir("tmpdir").join("test_mesh.vtk")
    pyvista.save_meshio(filename, mesh_in)
    mesh = pyvista.read_meshio(filename)

    # Assert mesh is still the same
    assert numpy.allclose(mesh_in.points, mesh.points)
    assert numpy.allclose(mesh_in.cells, mesh.cells)
    for k, v in mesh_in.point_arrays.items():
        assert numpy.allclose(v, mesh.point_arrays[k.replace(" ", "_")])
    for k, v in mesh_in.cell_arrays.items():
        assert numpy.allclose(v, mesh.cell_arrays[k.replace(" ", "_")])