import numpy as np

import pyvista
from pyvista import examples


def test_eq_wrong_type(sphere):
    assert sphere != [1, 2, 3]


def test_uniform_eq():
    orig = examples.load_uniform()
    copy = orig.copy(deep=True)
    copy.origin = [1, 1, 1]
    assert orig != copy

    copy.origin = [0, 0, 0]
    assert orig == copy

    copy.point_data.clear()
    assert orig != copy


def test_polydata_eq(sphere):
    sphere.clear_data()
    sphere.point_data['data0'] = np.zeros(sphere.n_points)
    sphere.point_data['data1'] = np.arange(sphere.n_points)

    copy = sphere.copy(deep=True)
    assert sphere == copy

    copy.faces = [3, 0, 1, 2]
    assert sphere != copy

    copy = sphere.copy(deep=True)
    copy.field_data['new'] = [1]
    assert sphere != copy

    copy = sphere.copy(deep=True)
    copy.point_data['new'] = range(sphere.n_points)
    assert sphere != copy

    copy = sphere.copy(deep=True)
    copy.cell_data['new'] = range(sphere.n_cells)
    assert sphere != copy

    copy = sphere.copy(deep=True)
    copy.point_data.active_scalars_name = 'data1'
    assert sphere != copy

    copy = sphere.copy(deep=True)
    copy.lines = [2, 0, 1]
    assert sphere != copy

    copy = sphere.copy(deep=True)
    copy.verts = [1, 0]
    assert sphere != copy


def test_unstructured_grid_eq(hexbeam):
    copy = hexbeam.copy()
    assert hexbeam == copy

    copy = hexbeam.copy()
    hexbeam.celltypes[0] = 0
    assert hexbeam != copy

    copy = hexbeam.copy()
    hexbeam.cell_connectivity[0] += 1
    assert hexbeam != copy


def test_metadata_save(hexbeam, tmpdir):
    """Test if complex and bool metadata is saved and restored."""
    filename = tmpdir.join('hexbeam.vtk')

    hexbeam.clear_data()
    point_data = np.arange(hexbeam.n_points)
    hexbeam.point_data['pt_data0'] = point_data + 1j * point_data

    bool_pt_data = np.zeros(hexbeam.n_points, dtype=bool)
    bool_pt_data[::2] = 1
    hexbeam.point_data['bool_data'] = bool_pt_data

    cell_data = np.arange(hexbeam.n_cells)
    bool_cell_data = np.zeros(hexbeam.n_cells, dtype=bool)
    bool_cell_data[::2] = 1
    hexbeam.cell_data['my_complex_cell_data'] = cell_data + 1j * cell_data
    hexbeam.cell_data['my_other_complex_cell_data'] = -cell_data - 1j * cell_data
    hexbeam.cell_data['bool_data'] = bool_cell_data

    # verify that complex data is restored
    hexbeam.save(filename)
    hexbeam_in = pyvista.read(filename)
    assert hexbeam_in.point_data['pt_data0'].dtype == np.complex128
    assert hexbeam_in.point_data['bool_data'].dtype == bool
    assert hexbeam_in.cell_data['my_complex_cell_data'].dtype == np.complex128
    assert hexbeam_in.cell_data['my_other_complex_cell_data'].dtype == np.complex128
    assert hexbeam_in.cell_data['bool_data'].dtype == bool

    # metadata should be removed from the field data
    assert not hexbeam_in.field_data
