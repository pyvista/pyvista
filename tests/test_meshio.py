import pathlib

import meshio
import numpy as np
import pytest

import pyvista as pv
from pyvista import examples

beam = pv.UnstructuredGrid(examples.hexbeamfile)
airplane = examples.load_airplane().cast_to_unstructured_grid()
uniform = examples.load_uniform().cast_to_unstructured_grid()
mesh2d = meshio.Mesh(
    points=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    cells=[("triangle", [[0, 1, 2], [1, 3, 2]])],
)


@pytest.mark.parametrize("mesh_in", [beam, airplane, uniform, mesh2d])
def test_meshio(mesh_in, tmpdir):
    if isinstance(mesh_in, meshio.Mesh):
        mesh_in = pv.from_meshio(mesh_in)

    # Save and read reference mesh using meshio
    filename = tmpdir.mkdir("tmpdir").join("test_mesh.vtk")
    pv.save_meshio(filename, mesh_in)
    mesh = pv.read_meshio(filename)

    # Assert mesh is still the same
    assert np.allclose(mesh_in.points, mesh.points)
    if (mesh_in.celltypes == 11).all():
        cells = mesh_in.cells.reshape((mesh_in.n_cells, 9))[:, [0, 1, 2, 4, 3, 5, 6, 8, 7]].ravel()
        assert np.allclose(cells, mesh.cells)
    else:
        assert np.allclose(mesh_in.cells, mesh.cells)
    for k, v in mesh_in.point_data.items():
        assert np.allclose(v, mesh.point_data[k.replace(" ", "_")])
    for k, v in mesh_in.cell_data.items():
        assert np.allclose(v, mesh.cell_data[k.replace(" ", "_")])


def test_pathlib_read_write(tmpdir, sphere):
    path = pathlib.Path(str(tmpdir.mkdir("tmpdir").join('tmp.vtk')))
    pv.save_meshio(path, sphere)
    assert path.is_file()

    mesh = pv.read_meshio(path)
    assert isinstance(mesh, pv.UnstructuredGrid)
    assert mesh.points.shape == sphere.points.shape


def test_file_format():
    from meshio._exceptions import ReadError, WriteError

    with pytest.raises(ReadError):
        _ = pv.read_meshio(examples.hexbeamfile, file_format="bar")

    with pytest.raises((KeyError, WriteError)):
        pv.save_meshio("foo.bar", beam, file_format="bar")

    with pytest.raises((KeyError, WriteError)):
        pv.save_meshio("foo.npy", beam, file_format="npy")
