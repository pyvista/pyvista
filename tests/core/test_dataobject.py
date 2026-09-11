from __future__ import annotations

from collections import UserDict
import copy
import gc
import json
import multiprocessing
import pickle
import re
import sys
from unittest.mock import patch

import numpy as np
import pytest

import pyvista as pv
from pyvista import examples
from pyvista.core import _vtk_utilities
from pyvista.core.dataobject import USER_DICT_KEY
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.utilities.arrays import _SerializedDictArray
from pyvista.core.utilities.writer import BaseWriter
from tests.vtk_backend_divergence import INT32_COMPRESSION


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
    connectivity = hexbeam.cell_connectivity.copy()
    connectivity[0] += 1
    hexbeam.cell_connectivity = connectivity
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


@pytest.mark.parametrize('file_ext', ['.vtm'])
def test_save_nested_multiblock_field_data(tmp_path, file_ext):
    filename = 'mesh' + file_ext
    nested = pv.MultiBlock()
    nested.field_data['foo'] = ['bar']
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


@pytest.mark.parametrize('data_object_type', [pv.PolyData, pv.MultiBlock])
def test_user_dict(data_object_type):
    data_object = data_object_type()
    assert USER_DICT_KEY not in data_object.field_data.keys()

    data_object.user_dict['abc'] = 123
    assert USER_DICT_KEY in data_object.field_data.keys()

    new_dict = dict(ham='eggs')
    data_object.user_dict = new_dict
    assert data_object.user_dict == new_dict
    assert data_object.field_data[USER_DICT_KEY].tolist() == [json.dumps(new_dict)]

    new_dict = UserDict(test='string')
    data_object.user_dict = new_dict
    assert data_object.user_dict == new_dict
    assert data_object.field_data[USER_DICT_KEY].tolist() == [json.dumps(new_dict.data)]

    match = (
        "User dict can only be set with type <class 'dict'> or <class 'collections.UserDict'>."
        "\nGot <class 'int'> instead."
    )
    with pytest.raises(TypeError, match=match):
        data_object.user_dict = 42


@pytest.mark.parametrize('data_object_type', [pv.PolyData, pv.MultiBlock])
@pytest.mark.parametrize('method', ['set_none', 'clear', 'clear_field_data'])
def test_user_dict_removal(data_object_type, method):
    data_object = data_object_type()

    def clear_user_dict():
        if method == 'clear':
            data_object.field_data.clear()
        elif method == 'clear_field_data':
            data_object.clear_field_data()
        elif method == 'set_none':
            data_object.user_dict = None
        else:  # pragma: no cover -- parametrize covers every case
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
    handle = data_object.user_dict
    assert handle == expected_dict

    # Clear it
    clear_user_dict()

    assert USER_DICT_KEY not in data_object.field_data.keys()
    assert data_object.user_dict == {}
    assert data_object.user_dict is handle
    assert actual_dict == expected_dict


@pytest.mark.parametrize(
    'value', [dict(a=0), ['list'], ('tuple', 1), 'string', 0, 1.1, True, None]
)
def test_user_dict_values(ant, value):
    ant.user_dict['key'] = value
    with pytest.raises(TypeError, match='not JSON serializable'):
        ant.user_dict['key'] = np.array(value)
    assert ant.user_dict['key'] == value
    with pytest.raises(TypeError, match='not JSON serializable'):
        ant.user_dict.update(other=1, key=np.array(value))
    assert 'other' not in ant.user_dict
    ant.user_dict['after'] = 1

    retrieved_value = json.loads(str(ant.user_dict))['key']

    # Round brackets '()' are saved as square brackets '[]' in JSON
    expected_value = list(value) if isinstance(value, tuple) else value
    assert retrieved_value == expected_value


def test_user_dict_repr(ant):
    ant.user_dict['foo'] = 'bar'
    user_dict = ant.user_dict
    assert repr(user_dict) == str(user_dict)


@pytest.mark.parametrize(
    ('make_data_object', 'ext'),
    [
        (lambda: pv.MultiBlock([examples.load_ant()]), '.vtm'),
        (examples.load_ant, '.vtp'),
        (examples.load_ant, '.vtkhdf'),
    ],
)
def test_user_dict_write_read(tmp_path, make_data_object, ext):
    if pv.vtk_version_info < (9, 4) and ext == '.vtkhdf':
        return  # can't use VTKHDF on VTK<9.4.0

    data_object = make_data_object()

    # test dict is restored after writing to file
    dict_data = dict(foo='bar')
    data_object.user_dict = dict_data

    assert USER_DICT_KEY in repr(data_object.field_data)

    filepath = tmp_path / ('data_object' + ext)
    data_object.save(filepath)

    data_object_read = pv.read(filepath)

    assert data_object_read.user_dict == dict_data
    assert USER_DICT_KEY in repr(data_object_read.field_data)


def _labelled_image():
    image = pv.ImageData(dimensions=(3, 3, 3))
    image['labels'] = np.arange(image.n_points) % 3
    return image


def _other_with_dict():
    other = pv.Cube()
    other.user_dict['name'] = 'other'
    return other


def _pickled(mesh):
    return pickle.loads(pickle.dumps(mesh))


USER_DICT_OPERATIONS = [
    pytest.param(pv.Sphere, lambda m: m.copy(), id='copy'),
    pytest.param(pv.Sphere, lambda m: m.copy(deep=False), id='shallow_copy'),
    pytest.param(pv.Sphere, _pickled, id='pickle'),
    pytest.param(pv.Sphere, lambda m: m.clip(), id='clip'),
    pytest.param(pv.Sphere, lambda m: m.slice(), id='slice'),
    pytest.param(pv.Sphere, lambda m: m.extract_surface(algorithm=None), id='extract_surface'),
    pytest.param(pv.Sphere, lambda m: m.triangulate(), id='triangulate'),
    pytest.param(pv.Sphere, lambda m: m.decimate(0.5), id='decimate'),
    pytest.param(pv.Sphere, lambda m: m.smooth(), id='smooth'),
    pytest.param(pv.Sphere, lambda m: m.subdivide(1), id='subdivide'),
    pytest.param(pv.Sphere, lambda m: m.compute_normals(), id='compute_normals'),
    pytest.param(pv.Sphere, lambda m: m.elevation(), id='elevation'),
    pytest.param(pv.Sphere, lambda m: m.elevation().warp_by_scalar(), id='warp_by_scalar'),
    pytest.param(pv.Sphere, lambda m: m.connectivity(), id='connectivity'),
    pytest.param(pv.Sphere, lambda m: m.extract_cells(range(10)), id='extract_cells'),
    pytest.param(pv.Sphere, lambda m: m.delaunay_2d(), id='delaunay_2d'),
    pytest.param(pv.Sphere, lambda m: m.clean(), id='clean'),
    pytest.param(pv.Sphere, lambda m: m.flip_faces(), id='flip_faces'),
    pytest.param(pv.Sphere, lambda m: m.reflect((1, 0, 0)), id='reflect'),
    pytest.param(pv.Sphere, lambda m: m.rotate_x(90), id='rotate_x'),
    pytest.param(pv.Sphere, lambda m: m.transform(np.eye(4), inplace=False), id='transform'),
    pytest.param(pv.Sphere, lambda m: m.sample(pv.Cube()), id='sample'),
    pytest.param(
        pv.Sphere, lambda m: m.cast_to_unstructured_grid(), id='cast_to_unstructured_grid'
    ),
    pytest.param(pv.Sphere, lambda m: m + _other_with_dict(), id='merge'),
    pytest.param(examples.load_uniform, lambda m: m.threshold(0.5), id='threshold'),
    pytest.param(examples.load_uniform, lambda m: m.contour(), id='contour'),
    pytest.param(examples.load_uniform, lambda m: m.gaussian_smooth(), id='gaussian_smooth'),
    pytest.param(examples.load_uniform, lambda m: m.points_to_cells(), id='points_to_cells'),
    pytest.param(examples.load_uniform, lambda m: m.cells_to_points(), id='cells_to_points'),
    pytest.param(
        examples.load_uniform, lambda m: m.cell_data_to_point_data(), id='cell_data_to_point_data'
    ),
    pytest.param(
        examples.load_uniform, lambda m: m.point_data_to_cell_data(), id='point_data_to_cell_data'
    ),
    pytest.param(
        examples.load_uniform, lambda m: m.extract_subset((0, 3, 0, 3, 0, 3)), id='extract_subset'
    ),
    pytest.param(_labelled_image, lambda m: m.pack_labels(), id='pack_labels'),
    pytest.param(lambda: pv.MultiBlock([pv.Sphere()]), lambda m: m.copy(), id='multiblock_copy'),
    pytest.param(
        lambda: pv.MultiBlock([pv.Sphere()]),
        lambda m: m.copy(deep=False),
        id='multiblock_shallow_copy',
    ),
    pytest.param(lambda: pv.MultiBlock([pv.Sphere()]), _pickled, id='multiblock_pickle'),
]


@pytest.mark.parametrize(('make_input', 'operation'), USER_DICT_OPERATIONS)
def test_user_dict_survives_operation(make_input, operation):
    mesh = make_input()
    handle = mesh.user_dict
    handle['name'] = 'input'

    out = operation(mesh)
    assert out is not mesh
    assert out.user_dict == {'name': 'input'}

    # The output owns its dict, whichever side is written first
    out.user_dict['out'] = 1
    assert mesh.user_dict == {'name': 'input'}
    handle['in'] = 2
    assert out.user_dict == {'name': 'input', 'out': 1}
    assert mesh.user_dict is handle


def test_user_dict_survives_source_deletion():
    mesh = pv.Sphere()
    mesh.user_dict['name'] = 'input'
    out = mesh.clip()
    del mesh
    gc.collect()
    assert out.user_dict == {'name': 'input'}
    out.user_dict['out'] = 1
    assert out.field_data[USER_DICT_KEY].tolist() == [json.dumps({'name': 'input', 'out': 1})]


@pytest.mark.parametrize(
    ('operation', 'expected'),
    [
        pytest.param(lambda m: m.clip(inplace=True), {'name': 'input'}, id='clip_inplace'),
        pytest.param(lambda m: m.point_data.clear(), {'name': 'input'}, id='point_data_clear'),
        pytest.param(lambda m: m.copy_from(pv.Cube()), {}, id='copy_from'),
        pytest.param(
            lambda m: m.copy_from(_other_with_dict()), {'name': 'other'}, id='copy_from_dict'
        ),
        pytest.param(
            lambda m: m.copy_from(_other_with_dict(), deep=False),
            {'name': 'other'},
            id='copy_from_dict_shallow',
        ),
        pytest.param(lambda m: m.clear_field_data(), {}, id='clear_field_data'),
        pytest.param(lambda m: m.field_data.clear(), {}, id='field_data_clear'),
        pytest.param(lambda m: m.field_data.remove(USER_DICT_KEY), {}, id='field_data_remove'),
        pytest.param(
            lambda m: m.field_data.__setitem__(USER_DICT_KEY, ['{"z": 9}']),
            {'z': 9},
            id='direct_write',
        ),
        pytest.param(lambda m: setattr(m, 'user_dict', None), {}, id='set_none'),
        pytest.param(lambda m: setattr(m, 'user_dict', {'x': 1}), {'x': 1}, id='set_dict'),
    ],
)
def test_user_dict_handle_persists(operation, expected):
    mesh = pv.Sphere()
    handle = mesh.user_dict
    handle['name'] = 'input'

    operation(mesh)
    assert mesh.user_dict is handle
    assert handle == expected

    handle['later'] = 1
    expected = {**expected, 'later': 1}
    assert mesh.user_dict == expected
    assert mesh.field_data[USER_DICT_KEY].tolist() == [json.dumps(expected)]


def test_user_dict_read_does_not_add_field_data():
    mesh = pv.Sphere()
    assert mesh.user_dict == {}
    assert 'name' not in mesh.user_dict
    assert USER_DICT_KEY not in mesh.field_data
    assert mesh == pv.Sphere()

    mesh.user_dict['name'] = 'input'
    assert USER_DICT_KEY in mesh.field_data


def test_user_dict_setter_copies_input():
    source = {'a': 1}
    mesh = pv.Sphere()
    mesh.user_dict = source
    mesh.user_dict['b'] = 2
    assert source == {'a': 1}
    source['c'] = 3
    assert mesh.user_dict == {'a': 1, 'b': 2}


@pytest.mark.parametrize(
    'write',
    [
        pytest.param(lambda d, k: d.__setitem__(k, 'x'), id='setitem'),
        pytest.param(lambda d, k: d.__setitem__('a', {k: 'x'}), id='nested'),
        pytest.param(lambda d, k: d.__setitem__('a', [{k: 'x'}]), id='nested_in_list'),
        pytest.param(lambda d, k: d.update({k: 'x'}), id='update'),
        pytest.param(lambda d, k: d.setdefault(k, 'x'), id='setdefault'),
        pytest.param(lambda d, k: setattr(d, 'data', {k: 'x'}), id='data'),
    ],
)
@pytest.mark.parametrize('key', [1, 1.5, True, None])
def test_user_dict_non_string_key_deprecated(write, key):
    mesh = pv.Sphere()
    match = (
        f'The user_dict key {key!r} is not a string, which is deprecated. '
        f'JSON stores keys as strings, so use {json.dumps(key)!r} instead.'
    )
    with pytest.warns(PyVistaDeprecationWarning, match=re.escape(match)):
        write(mesh.user_dict, key)
    assert f'"{json.dumps(key)}"' in str(mesh.user_dict)


def test_user_dict_setter_non_string_key_deprecated():
    mesh = pv.Sphere()
    with pytest.warns(PyVistaDeprecationWarning, match='is not a string'):
        mesh.user_dict = {1: 'x'}
    assert mesh.user_dict == {1: 'x'}
    assert mesh.copy().user_dict == {'1': 'x'}


def test_user_dict_keys_must_be_json_keys():
    mesh = pv.Sphere()
    with pytest.raises(TypeError, match='keys must be str, int, float, bool or None'):
        mesh.user_dict[(1, 2)] = 'x'
    assert mesh.user_dict == {}


@pytest.mark.parametrize(
    'make_copy', [copy.copy, copy.deepcopy, lambda d: d.copy()], ids=['copy', 'deepcopy', 'method']
)
def test_user_dict_copy_is_detached(make_copy):
    mesh = pv.Sphere()
    mesh.user_dict['name'] = 'input'
    copied = make_copy(mesh.user_dict)
    assert isinstance(copied, _SerializedDictArray)
    assert copied == {'name': 'input'}
    assert str(copied) == str(mesh.user_dict)

    copied['copy'] = 1
    mesh.user_dict['mesh'] = 2
    assert mesh.user_dict == {'name': 'input', 'mesh': 2}
    assert copied == {'name': 'input', 'copy': 1}
    assert USER_DICT_KEY in mesh.field_data
    assert mesh.field_data.VTKObject.GetAbstractArray(USER_DICT_KEY) is mesh.user_dict


def test_user_dict_dict_api():
    mesh = pv.Sphere()
    user_dict = mesh.user_dict

    def check(expected):
        assert user_dict == expected
        assert str(user_dict) == json.dumps(expected)
        assert mesh.field_data[USER_DICT_KEY].tolist() == [json.dumps(expected)]

    user_dict.update({'a': 1}, b=2)
    check({'a': 1, 'b': 2})
    user_dict.update([('c', 3)])
    check({'a': 1, 'b': 2, 'c': 3})
    user_dict |= {'d': 4}
    check({'a': 1, 'b': 2, 'c': 3, 'd': 4})

    assert user_dict.pop('d') == 4
    assert user_dict.pop('missing', None) is None
    with pytest.raises(KeyError):
        user_dict.pop('missing')
    check({'a': 1, 'b': 2, 'c': 3})

    assert user_dict.setdefault('e', 5) == 5
    assert user_dict.setdefault('e', 6) == 5
    check({'a': 1, 'b': 2, 'c': 3, 'e': 5})

    assert user_dict.popitem() == ('a', 1)
    del user_dict['b']
    check({'c': 3, 'e': 5})

    merged = user_dict | {'f': 6}
    assert isinstance(merged, _SerializedDictArray)
    assert merged == {'c': 3, 'e': 5, 'f': 6}
    check({'c': 3, 'e': 5})

    user_dict.clear()
    check({})


def test_user_dict_serializes_once_per_call(monkeypatch):
    calls = []
    original = _SerializedDictArray._serialize

    def counting(self):
        calls.append(1)
        original(self)

    monkeypatch.setattr(_SerializedDictArray, '_serialize', counting)
    mesh = pv.Sphere()
    user_dict = mesh.user_dict
    calls.clear()

    user_dict.update({f'k{i}': i for i in range(20)})
    assert len(calls) == 1
    calls.clear()
    user_dict.clear()
    assert len(calls) == 1
    calls.clear()
    mesh.user_dict = {'a': 1}
    assert len(calls) == 1


def test_default_pickle_format():
    assert pv.PICKLE_FORMAT == 'vtk'


@pytest.mark.parametrize('pickle_format', ['vtk', 'xml', 'legacy'])
def test_pickle_serialize_deserialize(datasets_no_pointset, pickle_format, capfd):
    """Test in-memory pickle protocol (multiprocessing/dask use case).

    Pickle is NOT a supported mesh file format — only the in-memory
    pickle protocol via ``__getstate__``/``__setstate__`` is tested
    here. File-format refusal is covered in ``test_reader.py``.
    """
    pv.set_pickle_format(pickle_format)
    for dataset in datasets_no_pointset:
        # These datasets carry no field data of their own.
        dataset.field_data['pickled_field'] = [1, 2, 3]
        dataset_2 = pickle.loads(pickle.dumps(dataset))
        assert not re.search(r'(WARN|ERR)\|', capfd.readouterr().err)

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


@pytest.mark.parametrize('pickle_format', ['vtk', 'xml', 'legacy'])
def test_pickle_drops_cached_vtk_objects(pickle_format):
    pv.set_pickle_format(pickle_format)
    mesh = pv.Sphere()
    # A bool array is tracked in the instance dict, which must survive the round trip
    mesh.point_data['flags'] = np.ones(mesh.n_points, dtype=bool)
    mesh.find_closest_point((0.0, 0.0, 0.0))
    assert isinstance(vars(mesh)['_point_locator'], pv._vtk.vtkObjectBase)
    assert mesh.points.dataset.Get() is mesh

    unpickled = pickle.loads(pickle.dumps(mesh))
    assert not any(isinstance(value, pv._vtk.vtkObjectBase) for value in vars(unpickled).values())
    assert unpickled == mesh
    assert unpickled.point_data['flags'].dtype == np.bool_
    assert unpickled.points.dataset.Get() is unpickled


def n_points(dataset):
    # used in multiprocessing test
    return dataset.n_points


@pytest.mark.parametrize('pickle_format', ['vtk', 'xml', 'legacy'])
def test_pickle_multiprocessing(datasets_no_pointset, pickle_format):
    # exercise pickling via multiprocessing
    pv.set_pickle_format(pickle_format)
    with multiprocessing.Pool(2) as p:
        res = p.map(n_points, datasets_no_pointset)
    for r, dataset in zip(res, datasets_no_pointset, strict=True):
        assert r == dataset.n_points


@pytest.mark.parametrize('pickle_format', ['vtk', 'xml', 'legacy'])
def test_pickle_multiblock(multiblock_all_no_pointset_with_nested_and_none, pickle_format):
    pv.set_pickle_format(pickle_format)
    multiblock = multiblock_all_no_pointset_with_nested_and_none

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
    pv.set_pickle_format(pickle_format)
    user_dict = {'custom_attribute': 42}
    sphere.user_dict = user_dict

    pickled = pickle.dumps(sphere)
    unpickled = pickle.loads(pickled)

    assert unpickled.user_dict == user_dict


@pytest.mark.parametrize('pickle_format', ['vtk', 'xml', 'legacy'])
def test_set_pickle_format(pickle_format):
    pv.set_pickle_format(pickle_format)
    assert pickle_format == pv.PICKLE_FORMAT


def test_pickle_invalid_format(sphere):
    match = 'Unsupported pickle format `invalid_format`.'
    with pytest.raises(ValueError, match=match):
        pv.set_pickle_format('invalid_format')

    pv.PICKLE_FORMAT = 'invalid_format'
    with pytest.raises(ValueError, match=match):
        pickle.dumps(sphere)


def test_pickle_deletes_cached_locators():
    poly = pv.Cone()

    for attr in ['_static_cell_locator', '_cell_tree_locator', '_point_locator']:
        # Access each locator to trigger the caching
        _ = getattr(poly, attr)

    pickle.loads(pickle.dumps(poly))


def test_save_raises_no_writers(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(pv.PolyData, '_WRITERS', None)
    match = re.escape(
        'PolyData writers are not specified, this should be a '
        'dict of (file extension: vtkWriter type)'
    )
    with pytest.raises(NotImplementedError, match=match):
        pv.Sphere().save('foo.vtp')


@pytest.mark.skip_vtk_backend('cvista', reason=INT32_COMPRESSION)
def test_save_compression(sphere, tmp_path):
    # int32 indices compress less, so pin the width the ratio below assumes.
    sphere = pv.PolyData(sphere.points, faces=sphere.faces.astype(np.int64))
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


def test_set_center(multiblock_all_with_nested_and_none):
    multi = multiblock_all_with_nested_and_none
    for mesh in [multi, *multi.recursive_iterator(skip_none=True)]:
        original_length = mesh.length
        new_center = (1.0, 2.0, 3.0)
        mesh.center = new_center
        actual_center = mesh.center
        assert np.allclose(actual_center, new_center), type(mesh)
        actual_length = mesh.length
        assert np.isclose(actual_length, original_length)


def test_raise_error_when_output_directory_is_missing(tmp_path):
    cylinder = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1))

    non_existent_dir = tmp_path / 'not_existing_directory'
    with pytest.raises(FileNotFoundError):
        cylinder.save(non_existent_dir / 'cylinder.vtk')

    with pytest.raises(FileNotFoundError):
        cylinder.cast_to_unstructured_grid().save(non_existent_dir / 'cylinder.vtu')


def test_raise_error_when_writing_is_failed(tmp_path):
    cylinder = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1))

    with patch.object(
        BaseWriter,
        'write',
        return_value=None,
    ):
        with pytest.raises(OSError, match='VTK writer failed to write file'):
            cylinder.save(tmp_path / 'cylinder.vtk')


def test_del_while_interpreter_is_finalizing(monkeypatch, sphere):
    """Deleting a dataset is a no-op once module globals have been cleared.

    ``isinstance`` against a cleared ``vtkObjectBase`` raises ``TypeError``,
    and an ``AttributeError`` raised from ``__getattribute__`` is silently
    rerouted into ``__getattr__``, which reaches ``importlib.metadata``.
    """
    sphere._glyph_geom = (pv.Sphere(),)
    # The interpreter calls the slot directly, so no attribute lookup is involved.
    finalizer = type(sphere).__del__

    monkeypatch.setattr(pv._vtk, 'vtkObjectBase', None)
    monkeypatch.setattr(_vtk_utilities, 'DisableVtkSnakeCase', None)
    monkeypatch.setattr(sys, 'is_finalizing', lambda: True)

    finalizer(sphere)
