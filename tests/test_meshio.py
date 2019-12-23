import numpy as np
import sys
import pytest

import pyvista
from pyvista import examples

TO_TEST = sys.version_info.major == 3 and sys.version_info.minor == 5
REASON = (
    "See https://github.com/pyvista/pyvista/pull/495 for meshio issues on Python 3.5"
)

beam = pyvista.UnstructuredGrid(examples.hexbeamfile)
airplane = examples.load_airplane().cast_to_unstructured_grid()
uniform = examples.load_uniform().cast_to_unstructured_grid()


@pytest.mark.parametrize("mesh_in", [beam, airplane, uniform])
@pytest.mark.skipif(TO_TEST, reason=REASON)
def test_meshio(mesh_in, tmpdir):
    # Save and read reference mesh using meshio
    filename = tmpdir.mkdir("tmpdir").join("test_mesh.vtk")
    pyvista.save_meshio(filename, mesh_in)
    mesh = pyvista.read_meshio(filename)

    # Assert mesh is still the same
    assert np.allclose(mesh_in.points, mesh.points)
    if (mesh_in.celltypes == 11).all():
        cells = mesh_in.cells.reshape((mesh_in.n_cells, 9))[
            :, [0, 1, 2, 4, 3, 5, 6, 8, 7]
        ].ravel()
        assert np.allclose(cells, mesh.cells)
    else:
        assert np.allclose(mesh_in.cells, mesh.cells)
    for k, v in mesh_in.point_arrays.items():
        assert np.allclose(v, mesh.point_arrays[k.replace(" ", "_")])
    for k, v in mesh_in.cell_arrays.items():
        assert np.allclose(v, mesh.cell_arrays[k.replace(" ", "_")])


@pytest.mark.skipif(TO_TEST, reason=REASON)
def test_file_format():
    from meshio._exceptions import ReadError

    with pytest.raises(ReadError):
        _ = pyvista.read_meshio(examples.hexbeamfile, file_format="bar")

    with pytest.raises(KeyError):
        pyvista.save_meshio("foo.bar", beam, file_format="bar")

    with pytest.raises(KeyError):
        pyvista.save_meshio("foo.npy", beam, file_format="npy")
