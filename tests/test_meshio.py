from __future__ import annotations

import pathlib

import meshio
import numpy as np
import pytest

import pyvista as pv
from pyvista import examples

cow = examples.download_cow().cast_to_unstructured_grid()
beam = pv.UnstructuredGrid(examples.hexbeamfile)
airplane = examples.load_airplane().cast_to_unstructured_grid()
uniform = examples.load_uniform().cast_to_unstructured_grid()
mesh2d = meshio.Mesh(
    points=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    cells=[("triangle", [[0, 1, 2], [1, 3, 2]])],
)
polyhedron = meshio.Mesh(
    points=[
        [0.3568221, -0.49112344, 0.79465446],
        [-0.3568221, -0.49112344, 0.79465446],
        [0.3568221, 0.49112344, -0.79465446],
        [-0.3568221, 0.49112344, -0.79465446],
        [0.0, 0.98224693, 0.18759243],
        [0.0, 0.60706198, 0.79465446],
        [0.0, -0.60706198, -0.79465446],
        [0.0, -0.98224693, -0.18759243],
        [0.93417233, 0.30353101, 0.18759247],
        [0.93417233, -0.30353101, -0.18759247],
        [-0.93417233, 0.30353101, 0.18759247],
        [-0.93417233, -0.30353101, -0.18759247],
        [-0.57735026, 0.18759249, 0.79465446],
        [0.57735026, -0.79465446, 0.18759249],
        [-0.57735026, -0.18759249, -0.79465446],
        [0.57735026, 0.79465446, -0.18759249],
        [0.57735026, 0.18759249, 0.79465446],
        [-0.57735026, 0.79465446, -0.18759249],
        [-0.57735026, -0.79465446, 0.18759249],
        [0.57735026, -0.18759249, -0.79465446],
        [0.3568221, 0.49112344, -1.0],
        [0.57735026, -0.18759249, -1.0],
        [0.0, -0.60706198, -1.0],
        [-0.57735026, -0.18759249, -1.0],
        [-0.3568221, 0.49112344, -1.0],
        [0.3568221, -0.49112344, 1.0],
        [0.57735026, 0.18759249, 1.0],
        [0.0, 0.60706198, 1.0],
        [-0.57735026, 0.18759249, 1.0],
        [-0.3568221, -0.49112344, 1.0],
    ],
    cells=[
        (
            "polyhedron20",
            [
                [
                    [0, 16, 5, 12, 1],
                    [1, 18, 7, 13, 0],
                    [2, 19, 6, 14, 3],
                    [3, 17, 4, 15, 2],
                    [4, 5, 16, 8, 15],
                    [5, 4, 17, 10, 12],
                    [6, 7, 18, 11, 14],
                    [7, 6, 19, 9, 13],
                    [8, 16, 0, 13, 9],
                    [9, 19, 2, 15, 8],
                    [10, 17, 3, 14, 11],
                    [11, 18, 1, 12, 10],
                ],
            ],
        ),
        (
            "polyhedron10",
            [
                [
                    [2, 19, 6, 14, 3],
                    [20, 21, 19, 2],
                    [21, 22, 6, 19],
                    [22, 23, 14, 6],
                    [23, 24, 3, 14],
                    [24, 20, 2, 3],
                    [20, 21, 22, 23, 24],
                ],
                [
                    [0, 16, 5, 12, 1],
                    [0, 16, 26, 25],
                    [16, 5, 27, 26],
                    [5, 12, 28, 27],
                    [12, 1, 29, 28],
                    [1, 0, 25, 29],
                    [25, 26, 27, 28, 29],
                ],
            ],
        ),
    ],
)


@pytest.mark.parametrize("mesh_in", [beam, airplane, uniform, mesh2d, polyhedron, cow])
def test_meshio(mesh_in, tmpdir):
    if isinstance(mesh_in, meshio.Mesh):
        mesh_in = pv.from_meshio(mesh_in)

    # Save and read reference mesh using meshio
    filename = tmpdir.mkdir("tmpdir").join("test_mesh.vtu")
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
    from meshio._exceptions import ReadError
    from meshio._exceptions import WriteError

    with pytest.raises(ReadError):
        _ = pv.read_meshio(examples.hexbeamfile, file_format="bar")

    with pytest.raises((KeyError, WriteError)):
        pv.save_meshio("foo.bar", beam, file_format="bar")

    with pytest.raises((KeyError, WriteError)):
        pv.save_meshio("foo.npy", beam, file_format="npy")
