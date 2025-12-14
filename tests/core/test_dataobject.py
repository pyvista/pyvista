from __future__ import annotations

from collections import UserDict
import json
import multiprocessing
import pickle
import re

import numpy as np
import pytest

import pyvista as pv
from pyvista import examples
from pyvista.core.dataobject import USER_DICT_KEY
from pyvista.core.utilities.fileio import save_pickle


def test_eq_wrong_type(sphere):
    assert sphere != [1, 2, 3]


def test_polydata_strip_neq():
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 3.0, 0.0],
            [0.0, 3.0, 0.0],
        ],
    )
    mesh1 = pv.PolyData(points, strips=(s := np.array([8, 0, 1, 2, 3, 4, 5, 6, 7])))

    s = s.copy()
    s[1:] = s[:0:-1]
    mesh2 = pv.PolyData(points, strips=s)

    assert mesh1 != mesh2

    s = s.copy()
    s[0] = 4
    mesh3 = pv.PolyData(points, strips=s[0:5])

    assert mesh1 != mesh3


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

    # test changing polyfaces is detected
    poly = examples.cells.Polyhedron()
    poly_copy = poly.copy()
    assert poly == poly_copy

    # we need to modify the face connectivity in-situ
    if pv.vtk_version_info < (9, 4):
        poly_faces = poly.GetFaces()
    else:
        poly_faces = poly_copy.GetPolyhedronFaces().GetConnectivityArray()
    pv.convert_array(poly_faces)[2] += 1
    assert poly != poly_copy

    # sanity check: ensure that modifying polyfaces doesn't change the
    # underlying cell connectivity
    assert np.allclose(poly.cell_connectivity, poly_copy.cell_connectivity)


def test_eq_nan_points():
    poly = pv.PolyData([np.nan, np.nan, np.nan])
    poly2 = poly.copy()
    assert poly == poly2


def test_eq_nan_array():
    poly = pv.PolyData()
    poly.field_data['data'] = [np.nan]
    poly2 = poly.copy()
    assert poly == poly2


def test_eq_string_array():
    poly = pv.PolyData()
    poly.field_data['data'] = ['abc']
    poly2 = poly.copy()
    assert poly == poly2


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


@pytest.mark.needs_vtk_version(9, 3)
@pytest.mark.parametrize('file_ext', ['.pkl', '.vtm'])
def test_save_nested_multiblock_field_data(tmp_path, file_ext):
    filename = 'mesh' + file_ext
    nested = pv.MultiBlock()
    nested.field_data['foo'] = 'bar'
    root = pv.MultiBlock([nested])

    # Save the multiblock and expect a warning
    match = (
        "Nested MultiBlock at index [0] with name 'Block-00' has field data "
        'which will not be saved.\n'
        'See https://gitlab.kitware.com/vtk/vtk/-/issues/19414 \n'
        'Use `move_nested_field_data_to_root` to store the field data with the root '
        'MultiBlock before saving.'
    )
    with pytest.warns(UserWarning, match=re.escape(match)):
        root.save(tmp_path / filename)

    # Check that the bug exists, and that the field data is not loaded
    loaded = pv.read(root)
    assert loaded[0].field_data.keys() == []

    # Save again without field data, no warning is emitted
    nested.clear_field_data()
    root.save(tmp_path / filename)


@pytest.mark.parametrize('data_object', [pv.PolyData(), pv.MultiBlock()])
def test_user_dict(data_object):
    assert USER_DICT_KEY not in data_object.field_data.keys()

    data_object.user_dict['abc'] = 123
    assert USER_DICT_KEY in data_object.field_data.keys()

    new_dict = dict(ham='eggs')
    data_object.user_dict = new_dict
    assert data_object.user_dict == new_dict
    assert data_object.field_data[USER_DICT_KEY] == json.dumps(new_dict)

    new_dict = UserDict(test='string')
    data_object.user_dict = new_dict
    assert data_object.user_dict == new_dict
    assert data_object.field_data[USER_DICT_KEY] == json.dumps(new_dict.data)

    match = (
        "User dict can only be set with type <class 'dict'> or <class 'collections.UserDict'>."
        "\nGot <class 'int'> instead."
    )
    with pytest.raises(TypeError, match=match):
        data_object.user_dict = 42


@pytest.mark.parametrize('data_object', [pv.PolyData(), pv.MultiBlock()])
@pytest.mark.parametrize('method', ['set_none', 'clear', 'clear_field_data'])
def test_user_dict_removal(data_object, method):
    def clear_user_dict():
        if method == 'clear':
            data_object.field_data.clear()
        elif method == 'clear_field_data':
            data_object.clear_field_data()
        elif method == 'set_none':
            data_object.user_dict = None
        else:
            msg = f'Invalid test method {method}.'
            raise RuntimeError(msg)

    # Clear before and after to ensure full test coverage of branches
    clear_user_dict()

    # Create dict for test and copy it since we want to test that the source dict itself
    # isn't cleared when clearing the user_dict
    expected_dict = dict(a=0)
    actual_dict = expected_dict.copy()

    # Set user dict
    data_object.user_dict = actual_dict
    assert data_object.user_dict == expected_dict

    # Clear it
    clear_user_dict()

    assert USER_DICT_KEY not in data_object.field_data.keys()
    assert data_object.user_dict == {}
    assert actual_dict == expected_dict


@pytest.mark.parametrize(
    'value', [dict(a=0), ['list'], ('tuple', 1), 'string', 0, 1.1, True, None]
)
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
    [
        (pv.MultiBlock([examples.load_ant()]), '.vtm'),
        (examples.load_ant(), '.vtp'),
        (examples.load_ant(), '.vtkhdf'),
    ],
)
def test_user_dict_write_read(tmp_path, data_object, ext):
    if pv.vtk_version_info < (9, 4) and ext == '.vtkhdf':
        return  # can't use VTKHDF on VTK<9.4.0

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
    assert merged.user_dict['name'] == 'sphere1'


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


@pytest.mark.parametrize('pickle_format', ['vtk', 'xml', 'legacy'])
@pytest.mark.parametrize('file_ext', ['.pkl', '.pickle', '', None])
def test_pickle_serialize_deserialize(datasets, pickle_format, file_ext, tmp_path):
    if pickle_format == 'vtk' and pv.vtk_version_info < (9, 3):
        pytest.xfail('VTK version not supported.')

    pv.set_pickle_format(pickle_format)
    for dataset in datasets:
        if file_ext is not None:
            filepath_save = tmp_path / ('data_object' + file_ext)
            if file_ext == '':
                save_pickle(filepath_save, dataset)
                filepath_read = tmp_path / ('data_object' + '.pkl')
            else:
                dataset.save(filepath_save)
                filepath_read = filepath_save
            dataset_2 = pv.read(filepath_read)
        else:
            dataset_2 = pickle.loads(pickle.dumps(dataset))

        # check python attributes are the same
        for attr in dataset.__dict__:
            assert getattr(dataset_2, attr) == getattr(dataset, attr)

        # check data is the same:
        assert dataset_2 == dataset

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


@pytest.mark.parametrize('pickle_format', ['vtk', 'xml', 'legacy'])
def test_pickle_multiprocessing(datasets, pickle_format):
    if pickle_format == 'vtk' and pv.vtk_version_info < (9, 3):
        pytest.xfail('VTK version not supported.')

    # exercise pickling via multiprocessing
    pv.set_pickle_format(pickle_format)
    with multiprocessing.Pool(2) as p:
        res = p.map(n_points, datasets)
    for r, dataset in zip(res, datasets, strict=True):
        assert r == dataset.n_points


@pytest.mark.parametrize('pickle_format', ['vtk', 'xml', 'legacy'])
def test_pickle_multiblock(multiblock_all_with_nested_and_none, pickle_format):
    if pickle_format == 'vtk' and pv.vtk_version_info < (9, 3):
        pytest.xfail('VTK version not supported.')

    pv.set_pickle_format(pickle_format)
    multiblock = multiblock_all_with_nested_and_none

    if pickle_format in ['legacy', 'xml']:
        match = (
            "MultiBlock is not supported with 'xml' or 'legacy' pickle formats.\n"
            "Use `pyvista.PICKLE_FORMAT='vtk'`."
        )
        with pytest.raises(TypeError, match=match):
            pickle.dumps(multiblock)
    else:
        pickled = pickle.dumps(multiblock)
        assert isinstance(pickled, bytes)
        unpickled = pickle.loads(pickled)
        assert unpickled == multiblock


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


@pytest.mark.parametrize('pickle_format', ['vtk', 'xml', 'legacy'])
def test_set_pickle_format(pickle_format):
    if pickle_format == 'vtk' and pv.vtk_version_info < (9, 3):
        match = 'requires VTK >= 9.3'
        with pytest.raises(ValueError, match=match):
            pv.set_pickle_format(pickle_format)
    else:
        pv.set_pickle_format(pickle_format)
        assert pickle_format == pv.PICKLE_FORMAT


def test_pickle_invalid_format(sphere):
    match = 'Unsupported pickle format `invalid_format`.'
    with pytest.raises(ValueError, match=match):
        pv.set_pickle_format('invalid_format')

    pv.PICKLE_FORMAT = 'invalid_format'
    with pytest.raises(ValueError, match=match):
        pickle.dumps(sphere)


def test_save_raises_no_writers(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pv.PolyData, '_WRITERS', None)
    match = re.escape(
        'PolyData writers are not specified, this should be a '
        'dict of (file extension: vtkWriter type)'
    )
    with pytest.raises(NotImplementedError, match=match):
        pv.Sphere().save('foo.vtp')


def test_save_compression(sphere, tmp_path):
    path = tmp_path / 'tmp.vtp'
    sphere.save(path, compression='zlib')
    compressed_size = path.stat().st_size
    sphere.save(path, compression=None)
    uncompressed_size = path.stat().st_size
    assert compressed_size < (uncompressed_size / 4)


def test_is_empty(ant):
    assert pv.MultiBlock().is_empty
    assert not pv.MultiBlock([ant]).is_empty

    assert pv.PolyData().is_empty
    assert not ant.is_empty

    assert pv.Table().is_empty
    assert not pv.Table(dict(a=np.array([0]))).is_empty


def test_cast_to_multiblock(multiblock_all):
    partitioned = pv.PartitionedDataSet()
    multiblock = pv.MultiBlock()
    pointset = pv.PointSet()

    for block in [*multiblock_all, partitioned, multiblock, pointset]:
        multi = block.cast_to_multiblock()
        assert isinstance(multi, pv.MultiBlock)


def test_center(multiblock_all_with_nested_and_none):
    for block in multiblock_all_with_nested_and_none.recursive_iterator(skip_none=True):
        original_length = block.length
        new_center = (1.1, 2.2, 3.3)
        block.center = new_center
        actual_center = block.center
        assert np.allclose(actual_center, new_center)
        actual_length = block.length
        assert np.isclose(actual_length, original_length)
