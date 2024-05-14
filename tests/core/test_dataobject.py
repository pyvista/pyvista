from collections import UserDict
import json

import numpy as np
import pytest

import pyvista as pv
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
    hexbeam_in = pv.read(filename)
    assert hexbeam_in.point_data['pt_data0'].dtype == np.complex128
    assert hexbeam_in.point_data['bool_data'].dtype == bool
    assert hexbeam_in.cell_data['my_complex_cell_data'].dtype == np.complex128
    assert hexbeam_in.cell_data['my_other_complex_cell_data'].dtype == np.complex128
    assert hexbeam_in.cell_data['bool_data'].dtype == bool

    # metadata should be removed from the field data
    assert not hexbeam_in.field_data


@pytest.mark.parametrize('data_object', [pv.PolyData(), pv.MultiBlock()])
def test_user_dict(data_object):
    field_name = '_PYVISTA_USER_DICT'
    assert field_name not in data_object.field_data.keys()

    data_object.user_dict['abc'] = 123
    assert field_name in data_object.field_data.keys()

    new_dict = dict(ham='eggs')
    data_object.user_dict = new_dict
    assert data_object.user_dict == new_dict
    assert data_object.field_data[field_name] == json.dumps(new_dict)

    new_dict = UserDict(test='string')
    data_object.user_dict = new_dict
    assert data_object.user_dict == new_dict
    assert data_object.field_data[field_name] == json.dumps(new_dict.data)

    data_object.user_dict = None
    assert field_name not in data_object.field_data.keys()

    match = (
        "User dict can only be set with type <class 'dict'> or <class 'collections.UserDict'>."
        "\nGot <class 'int'> instead."
    )
    with pytest.raises(TypeError, match=match):
        data_object.user_dict = 42


@pytest.mark.parametrize('value', [dict(a=0), ['list'], ('tuple', 1), 'string', 0, 1.1, True, None])
def test_user_dict_values(ant, value):
    ant.user_dict['key'] = value
    with pytest.raises(TypeError, match='not JSON serializable'):
        ant.user_dict['key'] = np.array(value)

    retrieved_value = json.loads(repr(ant.user_dict))['key']

    # Round brackets '()' are saved as square brackets '[]' in JSON
    expected_value = list(value) if isinstance(value, tuple) else value
    assert retrieved_value == expected_value


@pytest.mark.parametrize(
    ('data_object', 'ext'),
    [(pv.MultiBlock([examples.load_ant()]), '.vtm'), (examples.load_ant(), '.vtp')],
)
def test_user_dict_write_read(tmp_path, data_object, ext):
    # test dict is restored after writing to file
    dict_data = dict(foo='bar')
    data_object.user_dict = dict_data

    dict_field_repr = repr(data_object.user_dict)
    field_data_repr = repr(data_object.field_data)
    assert dict_field_repr in field_data_repr

    filepath = tmp_path / ('data_object' + ext)
    data_object.save(filepath)

    data_object_read = pv.read(filepath)

    assert data_object_read.user_dict == dict_data

    dict_field_repr = repr(data_object.user_dict)
    field_data_repr = repr(data_object.field_data)
    assert dict_field_repr in field_data_repr


def test_user_dict_persists_with_merge_filter():
    sphere1 = pv.Sphere()
    sphere1.user_dict['name'] = 'sphere1'

    sphere2 = pv.Sphere()
    sphere2.user_dict['name'] = 'sphere2'

    merged = sphere1 + sphere2
    assert merged.user_dict['name'] == 'sphere2'


def test_user_dict_persists_with_threshold_filter(uniform):
    uniform.user_dict['name'] = 'uniform'
    uniform = uniform.threshold(0.5)
    assert uniform.user_dict['name'] == 'uniform'


def test_user_dict_persists_with_pack_labels_filter():
    image = pv.ImageData(dimensions=(2, 2, 2))
    image['labels'] = [0, 3, 3, 3, 3, 0, 2, 2]
    image.user_dict['name'] = 'image'
    image = image.pack_labels()
    assert image.user_dict['name'] == 'image'


def test_user_dict_persists_with_point_voxels_to_cell_voxels(uniform):
    uniform.user_dict['name'] = 'image'
    uniform.cells_to_points()
    assert uniform.user_dict['name'] == 'image'


def test_user_dict_persists_with_cell_voxels_to_point_voxels(uniform):
    uniform.user_dict['name'] = 'image'
    uniform.points_to_cells()
    assert uniform.user_dict['name'] == 'image'
