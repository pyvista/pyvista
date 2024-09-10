from __future__ import annotations

from collections import UserDict
import json
import multiprocessing
import pickle

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


def test_user_dict_persists_with_points_to_cells(uniform):
    uniform.user_dict['name'] = 'image'
    uniform.cells_to_points()
    assert uniform.user_dict['name'] == 'image'


def test_user_dict_persists_with_cells_to_points(uniform):
    uniform.user_dict['name'] = 'image'
    uniform.points_to_cells()
    assert uniform.user_dict['name'] == 'image'


def test_default_pickle_format():
    assert pv.PICKLE_FORMAT == 'vtk' if pv.vtk_version_info >= (9, 3) else 'xml'


@pytest.fixture
def _modifies_pickle_format():
    before = pv.PICKLE_FORMAT
    yield
    pv.PICKLE_FORMAT = before


@pytest.mark.usefixtures('_modifies_pickle_format')
@pytest.mark.parametrize('pickle_format', ['vtk', 'xml', 'legacy'])
def test_pickle_serialize_deserialize(datasets, pickle_format):
    if pickle_format == 'vtk' and pv.vtk_version_info < (9, 3):
        pytest.xfail('VTK version not supported.')

    pv.set_pickle_format(pickle_format)
    for dataset in datasets:
        dataset_2 = pickle.loads(pickle.dumps(dataset))

        # check python attributes are the same
        for attr in dataset.__dict__:
            assert getattr(dataset_2, attr) == getattr(dataset, attr)

        # check data is the same
        for attr in ('n_cells', 'n_points', 'n_arrays'):
            if hasattr(dataset, attr):
                assert getattr(dataset_2, attr) == getattr(dataset, attr)

        for attr in ('cells', 'points'):
            if hasattr(dataset, attr):
                arr_have = getattr(dataset_2, attr)
                arr_expected = getattr(dataset, attr)
                assert arr_have == pytest.approx(arr_expected)

        for name in dataset.point_data:
            arr_have = dataset_2.point_data[name]
            arr_expected = dataset.point_data[name]
            assert arr_have == pytest.approx(arr_expected)

        for name in dataset.cell_data:
            arr_have = dataset_2.cell_data[name]
            arr_expected = dataset.cell_data[name]
            assert arr_have == pytest.approx(arr_expected)

        for name in dataset.field_data:
            arr_have = dataset_2.field_data[name]
            arr_expected = dataset.field_data[name]
            assert arr_have == pytest.approx(arr_expected)


def n_points(dataset):
    # used in multiprocessing test
    return dataset.n_points


@pytest.mark.usefixtures('_modifies_pickle_format')
@pytest.mark.parametrize('pickle_format', ['vtk', 'xml', 'legacy'])
def test_pickle_multiprocessing(datasets, pickle_format):
    if pickle_format == 'vtk' and pv.vtk_version_info < (9, 3):
        pytest.xfail('VTK version not supported.')

    # exercise pickling via multiprocessing
    pv.set_pickle_format(pickle_format)
    with multiprocessing.Pool(2) as p:
        res = p.map(n_points, datasets)
    for r, dataset in zip(res, datasets):
        assert r == dataset.n_points


@pytest.mark.usefixtures('_modifies_pickle_format')
@pytest.mark.parametrize('pickle_format', ['vtk', 'xml', 'legacy'])
def test_pickle_multiblock(multiblock_all_with_nested_and_none, pickle_format):
    if pickle_format == 'vtk' and pv.vtk_version_info < (9, 3):
        pytest.xfail('VTK version not supported.')

    pv.set_pickle_format(pickle_format)
    multiblock = multiblock_all_with_nested_and_none

    if pickle_format in ['legacy', 'xml']:
        match = "MultiBlock is not supported with 'xml' or 'legacy' pickle formats.\nUse `pyvista.PICKLE_FORMAT='vtk'`."
        with pytest.raises(TypeError, match=match):
            pickle.dumps(multiblock)
    else:
        pickled = pickle.dumps(multiblock)
        assert isinstance(pickled, bytes)
        unpickled = pickle.loads(pickled)
        assert unpickled == multiblock


@pytest.mark.usefixtures('_modifies_pickle_format')
@pytest.mark.parametrize('pickle_format', ['vtk', 'xml', 'legacy'])
def test_pickle_user_dict(sphere, pickle_format):
    if pickle_format == 'vtk' and pv.vtk_version_info < (9, 3):
        pytest.xfail('VTK version not supported.')

    pv.set_pickle_format(pickle_format)
    user_dict = {'custom_attribute': 42}
    sphere.user_dict = user_dict

    pickled = pickle.dumps(sphere)
    unpickled = pickle.loads(pickled)

    assert unpickled.user_dict == user_dict


@pytest.mark.usefixtures('_modifies_pickle_format')
@pytest.mark.parametrize('pickle_format', ['vtk', 'xml', 'legacy'])
def test_set_pickle_format(pickle_format):
    if pickle_format == 'vtk' and pv.vtk_version_info < (9, 3):
        match = 'requires VTK >= 9.3'
        with pytest.raises(ValueError, match=match):
            pv.set_pickle_format(pickle_format)
    else:
        pv.set_pickle_format(pickle_format)
        assert pickle_format == pv.PICKLE_FORMAT


@pytest.mark.usefixtures('_modifies_pickle_format')
def test_pickle_invalid_format(sphere):
    match = 'Unsupported pickle format `invalid_format`.'
    with pytest.raises(ValueError, match=match):
        pv.set_pickle_format('invalid_format')

    pv.PICKLE_FORMAT = 'invalid_format'
    with pytest.raises(ValueError, match=match):
        pickle.dumps(sphere)
