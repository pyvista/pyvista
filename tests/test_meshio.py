from __future__ import annotations

import pathlib

import meshio
import numpy as np
import pytest

import pyvista as pv
from pyvista import examples


@pytest.fixture
def empty():
    return pv.UnstructuredGrid()


@pytest.fixture
def cow_ugrid():
    return examples.download_cow().cast_to_unstructured_grid()


@pytest.fixture
def points_only(cow_ugrid):
    cow_ugrid.cells = np.array((), dtype=int)
    assert cow_ugrid.n_cells == 0
    return cow_ugrid


@pytest.fixture
def airplane_ugrid(airplane):
    return airplane.cast_to_unstructured_grid()


@pytest.fixture
def uniform_ugrid(uniform):
    return uniform.cast_to_unstructured_grid()


@pytest.fixture
def uniform2d():
    return pv.ImageData(dimensions=(10, 10, 1)).cast_to_unstructured_grid()


@pytest.fixture
def hybrid():
    cells = [
        8, 0, 1, 2, 3, 4, 5, 6, 7,
        5, 4, 5, 6, 7, 8,
        4, 4, 8, 7, 9,
        4, 5, 6, 8, 10,
        6, 1, 11, 5, 2, 12, 6,
        6, 13, 0, 4, 14, 3, 7,
        30, 7,
        3, 5, 6, 10,
        3, 11, 12, 15,
        3, 5, 11, 15,
        3, 5, 15, 10,
        3, 6, 10, 15,
        3, 6, 15, 12,
        4, 11, 12, 6, 5,
        30, 7,
        3, 4, 7, 9,
        3, 13, 14, 16,
        3, 4, 16, 13,
        3, 4, 9, 16,
        3, 7, 9, 16,
        3, 7, 16, 14,
        4, 13, 14, 7, 4,
    ]  # fmt: skip
    return pv.UnstructuredGrid(
        cells,
        [
            pv.CellType.HEXAHEDRON,
            pv.CellType.PYRAMID,
            pv.CellType.TETRA,
            pv.CellType.TETRA,
            pv.CellType.WEDGE,
            pv.CellType.WEDGE,
            pv.CellType.POLYHEDRON,
            pv.CellType.POLYHEDRON,
        ],
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.5, 0.5, 1.5],
            [0.0, 0.5, 1.5],
            [1.0, 0.5, 1.5],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0],
            [2.0, 0.5, 1.0],
            [-1.0, 0.5, 1.0],
        ],
    )


@pytest.fixture
def quad_pixel_hex_voxel():
    return (
        examples.cells.Quadrilateral()
        + examples.cells.Pixel()
        + examples.cells.Hexahedron()
        + examples.cells.Voxel()
    )


@pytest.fixture
def mesh2d():
    return meshio.Mesh(
        points=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        cells=[('triangle', [[0, 1, 2], [1, 3, 2]])],
        cell_sets={'tri1': [[0]], 'tri2': [[1]]},
    )


@pytest.fixture
def polyhedron():
    return meshio.Mesh(
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
                'polyhedron20',
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
                'polyhedron10',
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


fixture_names = [
    'hexbeam',
    'airplane_ugrid',
    'uniform_ugrid',
    'uniform2d',
    'hybrid',
    'mesh2d',
    'polyhedron',
    'cow_ugrid',
    'empty',
    'points_only',
    'quad_pixel_hex_voxel',
]


@pytest.mark.parametrize('mesh_in_name', fixture_names)
def test_meshio(request, mesh_in_name):
    mesh_in = request.getfixturevalue(mesh_in_name)
    if isinstance(mesh_in, meshio.Mesh):
        mesh_in = pv.from_meshio(mesh_in)

    mesh = pv.from_meshio(pv.to_meshio(mesh_in))

    # Assert mesh is still the same
    assert np.allclose(mesh_in.points, mesh.points)
    if (mesh_in.celltypes == pv.CellType.PIXEL).all():
        cells = mesh_in.cells.reshape((mesh_in.n_cells, 5))[:, [0, 1, 2, 4, 3]].ravel()
        assert np.allclose(cells, mesh.cells)
    elif (mesh_in.celltypes == pv.CellType.VOXEL).all():
        cells = mesh_in.cells.reshape((mesh_in.n_cells, 9))[:, [0, 1, 2, 4, 3, 5, 6, 8, 7]].ravel()
        assert np.allclose(cells, mesh.cells)
    # Mixed cell types with voxels or pixels
    elif np.isin(mesh_in.celltypes, [pv.CellType.PIXEL, pv.CellType.VOXEL]).any():
        cells_in = []

        for cell_type, cell in mesh_in.cells_dict.items():
            for indices in cell:
                # Pixel
                if cell_type == pv.CellType.PIXEL:
                    cells_in.extend([len(indices), *indices[[0, 1, 3, 2]].tolist()])
                # Voxel
                elif cell_type == pv.CellType.VOXEL:
                    cells_in.extend([len(indices), *indices[[0, 1, 3, 2, 4, 5, 7, 6]].tolist()])
                else:
                    cells_in.extend([len(indices), *indices.tolist()])

        cells_out = [
            i
            for cell in mesh.cells_dict.values()
            for indices in cell
            for i in [len(indices), *indices.tolist()]
        ]

        assert cells_in == cells_out
    else:
        assert np.allclose(mesh_in.cells, mesh.cells)
    for k, v in mesh_in.point_data.items():
        assert np.allclose(v, mesh.point_data[k.replace(' ', '_')])
    for k, v in mesh_in.cell_data.items():
        assert np.allclose(v, mesh.cell_data[k.replace(' ', '_')])


def test_pathlib_read_write(tmpdir, sphere):
    path = pathlib.Path(str(tmpdir.mkdir('tmpdir').join('tmp.vtk')))
    pv.save_meshio(path, sphere)
    assert path.is_file()

    mesh = pv.read_meshio(path)
    assert isinstance(mesh, pv.UnstructuredGrid)
    assert mesh.points.shape == sphere.points.shape


def test_file_format(hexbeam):
    from meshio._exceptions import ReadError
    from meshio._exceptions import WriteError

    with pytest.raises(ReadError):
        _ = pv.read_meshio(examples.hexbeamfile, file_format='bar')

    with pytest.raises((KeyError, WriteError)):
        pv.save_meshio('foo.bar', hexbeam, file_format='bar')

    with pytest.raises((KeyError, WriteError)):
        pv.save_meshio('foo.npy', hexbeam, file_format='npy')
