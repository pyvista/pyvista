import numpy as np
import pytest

import pyvista as pv
from pyvista.utilities.misc import PyVistaDeprecationWarning


@pytest.fixture
def tiny_rectilinear():
    xrng = np.linspace(0, 3, 4)
    yrng = np.linspace(0, 3, 5)
    zrng = np.linspace(0, 3, 6)
    return pv.RectilinearGrid(xrng, yrng, zrng)


@pytest.mark.parametrize('tetra_per_cell', [5, 6, 12])
def test_to_tetrahedral(tiny_rectilinear, tetra_per_cell):
    tet_grid = tiny_rectilinear.to_tetrahedra(tetra_per_cell=tetra_per_cell)
    assert tet_grid.n_cells == tiny_rectilinear.n_cells * tetra_per_cell


def test_to_tetrahedral_raise(tiny_rectilinear):
    with pytest.raises(ValueError, match='either 5, 6, or 12'):
        tiny_rectilinear.to_tetrahedra(tetra_per_cell=9)


def test_to_tetrahedral_mixed(tiny_rectilinear):
    data = np.empty(tiny_rectilinear.n_cells, dtype=int)
    half = tiny_rectilinear.n_cells // 2
    data[:half] = 5
    data[half:] = 12
    tiny_rectilinear['data'] = data
    tet_grid = tiny_rectilinear.to_tetrahedra(mixed=True)
    assert tet_grid.n_cells == (half * 5 + half * 12)

    tet_grid = tiny_rectilinear.to_tetrahedra(mixed='data')
    assert tet_grid.n_cells == (half * 5 + half * 12)

    tet_grid = tiny_rectilinear.to_tetrahedra(mixed=data)
    assert tet_grid.n_cells == (half * 5 + half * 12)

    with pytest.raises(TypeError, match='mixed'):
        tet_grid = tiny_rectilinear.to_tetrahedra(mixed=123)


def test_to_tetrahedral_edge_case():
    with pytest.raises(RuntimeError, match='is 1'):
        pv.UniformGrid(dimensions=(1, 2, 2)).to_tetrahedra(tetra_per_cell=12)


def test_to_tetrahedral_pass_cell_ids(tiny_rectilinear):
    tet_grid = tiny_rectilinear.to_tetrahedra(pass_cell_ids=False, pass_data=False)
    assert not tet_grid.cell_data
    tet_grid = tiny_rectilinear.to_tetrahedra(pass_cell_ids=True, pass_data=False)
    assert 'vtkOriginalCellIds' in tet_grid.cell_data
    assert np.issubdtype(tet_grid.cell_data['vtkOriginalCellIds'].dtype, np.integer)


def test_to_tetrahedral_pass_cell_data(tiny_rectilinear):
    # test that data isn't passed
    tiny_rectilinear["cell_data"] = np.ones(tiny_rectilinear.n_cells)
    tiny_rectilinear["point_data"] = np.arange(tiny_rectilinear.n_points)
    tiny_rectilinear.set_active_scalars("cell_data")
    tet_grid = tiny_rectilinear.to_tetrahedra(pass_cell_ids=False, pass_data=False)
    assert not tet_grid.cell_data
    assert not tet_grid.point_data

    # test with cell data
    tet_grid = tiny_rectilinear.to_tetrahedra(pass_cell_ids=False, pass_data=True)
    assert tet_grid.cell_data
    assert "cell_data" in tet_grid.cell_data
    assert "point_data" in tet_grid.point_data
    assert tet_grid.active_scalars_name == "cell_data"

    # automatically added
    assert 'vtkOriginalCellIds' in tet_grid.cell_data

    with pytest.warns(PyVistaDeprecationWarning):
        tiny_rectilinear.to_tetrahedra(pass_cell_data=True)
        if pv._version.version_info >= (0, 43, 0):
            raise RuntimeError('Remove this deprecated kwarg')

    # Test point data active
    tiny_rectilinear.set_active_scalars("point_data")
    assert tiny_rectilinear.active_scalars_name == "point_data"
    tet_grid = tiny_rectilinear.to_tetrahedra(pass_cell_ids=False, pass_data=True)
    assert tet_grid.active_scalars_name == "point_data"
