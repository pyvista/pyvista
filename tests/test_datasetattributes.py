import os
import platform
from string import ascii_letters, digits, whitespace
import sys

from hypothesis import HealthCheck, given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import integers, lists, text
import numpy as np
from pytest import fixture, mark, raises

import pyvista
from pyvista.utilities import FieldAssociation

skip_windows = mark.skipif(os.name == 'nt', reason='Test fails on Windows')
skip_apple_silicon = mark.skipif(
    platform.system() == 'Darwin' and platform.processor() == 'arm',
    reason='Test fails on Apple Silicon',
)


@fixture()
def hexbeam_point_attributes(hexbeam):
    return hexbeam.point_data


@fixture()
def hexbeam_field_attributes(hexbeam):
    return hexbeam.field_data


@fixture()
def insert_arange_narray(hexbeam_point_attributes):
    n_points = hexbeam_point_attributes.dataset.GetNumberOfPoints()
    sample_array = np.arange(n_points)
    hexbeam_point_attributes.set_array(sample_array, 'sample_array')
    return hexbeam_point_attributes, sample_array


@fixture()
def insert_bool_array(hexbeam_point_attributes):
    n_points = hexbeam_point_attributes.dataset.GetNumberOfPoints()
    sample_array = np.ones(n_points, np.bool_)
    hexbeam_point_attributes.set_array(sample_array, 'sample_array')
    return hexbeam_point_attributes, sample_array


@fixture()
def insert_string_array(hexbeam_point_attributes):
    n_points = hexbeam_point_attributes.dataset.GetNumberOfPoints()
    sample_array = np.repeat("A", n_points)
    hexbeam_point_attributes.set_array(sample_array, 'sample_array')
    return hexbeam_point_attributes, sample_array


def test_init(hexbeam):
    attributes = pyvista.DataSetAttributes(
        hexbeam.GetPointData(), dataset=hexbeam, association=FieldAssociation.POINT
    )
    assert attributes.VTKObject == hexbeam.GetPointData()
    assert attributes.dataset == hexbeam
    assert attributes.association == FieldAssociation.POINT


def test_bool(hexbeam_point_attributes):
    assert bool(len(hexbeam_point_attributes)) is bool(hexbeam_point_attributes)
    hexbeam_point_attributes.clear()
    assert bool(len(hexbeam_point_attributes)) is bool(hexbeam_point_attributes)


def test_getitem(hexbeam_point_attributes):
    with raises(TypeError, match='Only strings'):
        hexbeam_point_attributes[0]


def test_setitem(hexbeam_point_attributes):
    with raises(TypeError, match='Only strings'):
        hexbeam_point_attributes[0]


def test_repr(hexbeam_point_attributes):
    repr_str = str(hexbeam_point_attributes)
    assert 'POINT' in repr_str
    assert 'DataSetAttributes' in repr_str
    assert 'Contains arrays' in repr_str
    assert '...' not in repr_str

    # ensure long names are abbreviated
    sz = hexbeam_point_attributes.values()[0].size
    data = np.zeros(sz)
    hexbeam_point_attributes['thisisaverylongnameover20char'] = data
    assert '...' in str(hexbeam_point_attributes)

    # ensure datatype str is in repr
    assert str(data.dtype) in str(hexbeam_point_attributes)

    # ensure VECTOR in repr
    vectors0 = np.random.random((sz, 3))
    hexbeam_point_attributes.set_vectors(vectors0, 'vectors0')
    assert 'VECTOR' in str(hexbeam_point_attributes)


def test_empty_active_vectors(hexbeam):
    assert hexbeam.active_vectors is None


def test_valid_array_len_points(hexbeam):
    assert hexbeam.point_data.valid_array_len == hexbeam.n_points


def test_valid_array_len_cells(hexbeam):
    assert hexbeam.cell_data.valid_array_len == hexbeam.n_cells


def test_valid_array_len_field(hexbeam):
    assert hexbeam.field_data.valid_array_len is None


def test_get(sphere):
    point_data = np.arange(sphere.n_points)
    sphere.clear_data()
    key = 'my-data'
    sphere.point_data[key] = point_data
    assert np.array_equal(sphere.point_data.get(key), point_data)
    assert sphere.point_data.get('invalid-key') is None

    default = 'default'
    assert sphere.point_data.get('invalid-key', default) is default


def test_active_scalars_name(sphere):
    sphere.clear_data()
    assert sphere.point_data.active_scalars_name is None

    key = 'data0'
    sphere.point_data[key] = range(sphere.n_points)
    assert sphere.point_data.active_scalars_name == key

    sphere.point_data.active_scalars_name = None
    assert sphere.point_data.active_scalars_name is None


def test_set_scalars(sphere):
    scalars = np.array(sphere.n_points)
    key = 'scalars'
    sphere.point_data.set_scalars(scalars, key)
    assert sphere.point_data.active_scalars_name == key


def test_eq(sphere):
    sphere = pyvista.Sphere()
    sphere.clear_data()

    # check wrong type
    assert sphere.point_data != [1, 2, 3]

    sphere.point_data['data0'] = np.zeros(sphere.n_points)
    sphere.point_data['data1'] = np.arange(sphere.n_points)
    deep_cp = sphere.copy(deep=True)
    shal_cp = sphere.copy(deep=False)
    assert sphere.point_data == deep_cp.point_data
    assert sphere.point_data == shal_cp.point_data

    # verify inplace change
    sphere.point_data['data0'] += 1
    assert sphere.point_data != deep_cp.point_data
    assert sphere.point_data == shal_cp.point_data

    # verify key removal
    deep_cp = sphere.copy(deep=True)
    del deep_cp.point_data['data0']
    assert sphere.point_data != deep_cp.point_data


def test_add_matrix(hexbeam):
    mat_shape = (hexbeam.n_points, 3, 2)
    mat = np.random.random(mat_shape)
    hexbeam.point_data.set_array(mat, 'mat')
    matout = hexbeam.point_data['mat'].reshape(mat_shape)
    assert np.allclose(mat, matout)


def test_set_active_scalars_fail(hexbeam):
    with raises(ValueError):
        hexbeam.set_active_scalars('foo', preference='field')
    with raises(KeyError):
        hexbeam.set_active_scalars('foo')


def test_set_active_vectors(hexbeam):
    vectors = np.random.random((hexbeam.n_points, 3))
    hexbeam['vectors'] = vectors
    hexbeam.set_active_vectors('vectors')
    assert np.allclose(hexbeam.active_vectors, vectors)


def test_set_vectors(hexbeam):
    assert hexbeam.point_data.active_vectors is None
    vectors = np.random.random((hexbeam.n_points, 3))
    hexbeam.point_data.set_vectors(vectors, 'my-vectors')
    assert np.allclose(hexbeam.point_data.active_vectors, vectors)

    # check clearing
    hexbeam.point_data.active_vectors_name = None
    assert hexbeam.point_data.active_vectors_name is None


def test_set_invalid_vectors(hexbeam):
    # verify non-vector data does not become active vectors
    not_vectors = np.random.random(hexbeam.n_points)
    with raises(ValueError):
        hexbeam.point_data.set_vectors(not_vectors, 'my-vectors')


def test_set_tcoords_name():
    mesh = pyvista.Cube()
    old_name = mesh.point_data.active_t_coords_name
    assert mesh.point_data.active_t_coords_name is not None
    mesh.point_data.active_t_coords_name = None
    assert mesh.point_data.active_t_coords_name is None

    mesh.point_data.active_t_coords_name = old_name
    assert mesh.point_data.active_t_coords_name == old_name


def test_set_bitarray(hexbeam):
    """Test bitarrays are properly loaded and represented in datasetattributes."""
    hexbeam.clear_data()
    assert 'bool' not in str(hexbeam.point_data)

    arr = np.zeros(hexbeam.n_points, dtype=bool)
    arr[::2] = 1
    hexbeam.point_data['bitarray'] = arr

    assert hexbeam.point_data['bitarray'].dtype == np.bool_
    assert 'bool' in str(hexbeam.point_data)
    assert np.allclose(hexbeam.point_data['bitarray'], arr)

    # ensure overwriting the type changes association
    hexbeam.point_data['bitarray'] = arr.astype(np.int32)
    assert hexbeam.point_data['bitarray'].dtype == np.int32


@mark.parametrize('array_key', ['invalid_array_name', -1])
def test_get_array_should_fail_if_does_not_exist(array_key, hexbeam_point_attributes):
    with raises(KeyError):
        hexbeam_point_attributes.get_array(array_key)


def test_get_array_should_return_bool_array(insert_bool_array):
    dsa, _ = insert_bool_array
    output_array = dsa.get_array('sample_array')
    assert output_array.dtype == np.bool_


def test_get_array_bool_array_should_be_identical(insert_bool_array):
    dsa, sample_array = insert_bool_array
    output_array = dsa.get_array('sample_array')
    assert np.array_equal(output_array, sample_array)


def test_add_should_not_add_none_array(hexbeam_point_attributes):
    with raises(TypeError):
        hexbeam_point_attributes.set_array(None, 'sample_array')


def test_add_should_contain_array_name(insert_arange_narray):
    dsa, _ = insert_arange_narray
    assert 'sample_array' in dsa


def test_add_should_contain_exact_array(insert_arange_narray):
    dsa, sample_array = insert_arange_narray
    assert np.array_equal(sample_array, dsa['sample_array'])


def test_getters_should_return_same_result(insert_arange_narray):
    dsa, sample_array = insert_arange_narray
    result_a = dsa.get_array('sample_array')
    result_b = dsa['sample_array']
    assert np.array_equal(result_a, result_b)


def test_contains_should_contain_when_added(insert_arange_narray):
    dsa, sample_array = insert_arange_narray
    assert 'sample_array' in dsa


def test_set_array_catch(hexbeam):
    data = np.zeros(hexbeam.n_points)
    with raises(TypeError, match='`name` must be a string'):
        hexbeam.point_data.set_array(data, name=['foo'])


@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(scalar=integers(min_value=-sys.maxsize - 1, max_value=sys.maxsize))
def test_set_array_should_accept_scalar_value(scalar, hexbeam_point_attributes):
    hexbeam_point_attributes.set_array(scalar, name='int_array')


@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(scalar=integers(min_value=-sys.maxsize - 1, max_value=sys.maxsize))
def test_set_array_scalar_value_should_give_array(scalar, hexbeam_point_attributes):
    hexbeam_point_attributes.set_array(scalar, name='int_array')
    expected = np.full(hexbeam_point_attributes.dataset.n_points, scalar)
    assert np.array_equal(expected, hexbeam_point_attributes['int_array'])


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(arr=lists(text(alphabet=ascii_letters + digits + whitespace), max_size=16))
def test_set_array_string_lists_should_equal(arr, hexbeam_field_attributes):
    hexbeam_field_attributes['string_arr'] = arr
    assert arr == hexbeam_field_attributes['string_arr'].tolist()


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(arr=arrays(dtype='U', shape=10))
def test_set_array_string_array_should_equal(arr, hexbeam_field_attributes):
    if not ''.join(arr).isascii():
        with raises(ValueError, match='non-ASCII'):
            hexbeam_field_attributes['string_arr'] = arr
        return

    hexbeam_field_attributes['string_arr'] = arr
    assert np.array_equiv(arr, hexbeam_field_attributes['string_arr'])


def test_hexbeam_field_attributes_active_scalars(hexbeam_field_attributes):
    with raises(TypeError):
        hexbeam_field_attributes.active_scalars


def test_should_remove_array(insert_arange_narray):
    dsa, sample_array = insert_arange_narray
    dsa.remove('sample_array')
    assert 'sample_array' not in dsa


def test_should_del_array(insert_arange_narray):
    dsa, sample_array = insert_arange_narray
    del dsa['sample_array']
    assert 'sample_array' not in dsa


def test_should_pop_array(insert_arange_narray):
    dsa, sample_array = insert_arange_narray
    dsa.pop('sample_array')
    assert 'sample_array' not in dsa


def test_pop_should_return_arange_narray(insert_arange_narray):
    dsa, sample_array = insert_arange_narray
    other_array = dsa.pop('sample_array')
    assert np.array_equal(other_array, sample_array)


def test_pop_should_return_bool_array(insert_bool_array):
    dsa, sample_array = insert_bool_array
    other_array = dsa.pop('sample_array')
    assert np.array_equal(other_array, sample_array)


def test_pop_should_return_string_array(insert_string_array):
    dsa, sample_array = insert_string_array
    other_array = dsa.pop('sample_array')
    assert np.array_equal(other_array, sample_array)


def test_should_pop_array_invalid(insert_arange_narray):
    dsa, sample_array = insert_arange_narray
    key = 'invalid_key'
    assert key not in dsa
    default = 20
    assert dsa.pop(key, default) is default


@mark.parametrize('removed_key', [None, 'nonexistent_array_name', '', -1])
def test_remove_should_fail_on_bad_argument(removed_key, hexbeam_point_attributes):
    if removed_key in [None, -1]:
        with raises(TypeError):
            hexbeam_point_attributes.remove(removed_key)
    else:
        with raises(KeyError):
            hexbeam_point_attributes.remove(removed_key)


@mark.parametrize('removed_key', [None, 'nonexistent_array_name', '', -1])
def test_del_should_fail_bad_argument(removed_key, hexbeam_point_attributes):
    if removed_key in [None, -1]:
        with raises(TypeError):
            del hexbeam_point_attributes[removed_key]
    else:
        with raises(KeyError):
            del hexbeam_point_attributes[removed_key]


@mark.parametrize('removed_key', [None, 'nonexistent_array_name', '', -1])
def test_pop_should_fail_bad_argument(removed_key, hexbeam_point_attributes):
    if removed_key in [None, -1]:
        with raises(TypeError):
            hexbeam_point_attributes.pop(removed_key)
    else:
        with raises(KeyError):
            hexbeam_point_attributes.pop(removed_key)


def test_length_should_increment_on_set_array(hexbeam_point_attributes):
    initial_len = len(hexbeam_point_attributes)
    n_points = hexbeam_point_attributes.dataset.GetNumberOfPoints()
    sample_array = np.arange(n_points)
    hexbeam_point_attributes.set_array(sample_array, 'sample_array')
    assert len(hexbeam_point_attributes) == initial_len + 1


def test_length_should_decrement_on_remove(insert_arange_narray):
    dsa, sample_array = insert_arange_narray
    initial_len = len(dsa)
    dsa.remove('sample_array')
    assert len(dsa) == initial_len - 1


def test_length_should_decrement_on_pop(insert_arange_narray):
    dsa, sample_array = insert_arange_narray
    initial_len = len(dsa)
    dsa.pop('sample_array')
    assert len(dsa) == initial_len - 1


def test_length_should_be_0_on_clear(insert_arange_narray):
    dsa, sample_array = insert_arange_narray
    assert len(dsa) != 0
    dsa.clear()
    assert len(dsa) == 0


def test_keys_should_be_strings(insert_arange_narray):
    dsa, sample_array = insert_arange_narray
    for name in dsa.keys():
        assert type(name) == str


def test_key_should_exist(insert_arange_narray):
    dsa, sample_array = insert_arange_narray
    assert 'sample_array' in dsa.keys()


def test_values_should_be_pyvista_ndarrays(insert_arange_narray):
    dsa, sample_array = insert_arange_narray
    for arr in dsa.values():
        assert type(arr) == pyvista.pyvista_ndarray


def test_value_should_exist(insert_arange_narray):
    dsa, sample_array = insert_arange_narray
    for arr in dsa.values():
        if np.array_equal(sample_array, arr):
            return
    raise AssertionError('Array not in values.')


def test_active_scalars_setter(hexbeam_point_attributes):
    dsa = hexbeam_point_attributes
    assert dsa.active_scalars is None

    dsa.active_scalars_name = 'sample_point_scalars'
    assert dsa.active_scalars is not None
    assert dsa.GetScalars().GetName() == 'sample_point_scalars'


def test_active_scalars_setter_no_override(hexbeam):
    # Test that adding new array does not override
    assert hexbeam.active_scalars_name == 'sample_cell_scalars'
    hexbeam.cell_data['test'] = np.arange(0, hexbeam.n_cells, dtype=int)
    assert hexbeam.active_scalars_name == 'sample_cell_scalars'


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(arr=arrays(dtype='U', shape=10))
def test_preserve_field_data_after_extract_cells(hexbeam, arr):
    if not ''.join(arr).isascii():
        with raises(ValueError, match='non-ASCII'):
            hexbeam.field_data["foo"] = arr
        return

    # https://github.com/pyvista/pyvista/pull/934
    hexbeam.field_data["foo"] = arr
    extracted = hexbeam.extract_cells([0, 1, 2, 3])
    assert "foo" in extracted.field_data


def test_assign_labels_to_points(hexbeam):
    hexbeam.point_data.clear()
    labels = [f"Label {i}" for i in range(hexbeam.n_points)]
    hexbeam['labels'] = labels
    assert (hexbeam['labels'] == labels).all()


def test_normals_get(plane):
    plane.clear_data()
    assert plane.point_data.active_normals is None

    plane_w_normals = plane.compute_normals()
    assert np.array_equal(plane_w_normals.point_data.active_normals, plane_w_normals.point_normals)

    plane.point_data.active_normals_name = None
    assert plane.point_data.active_normals_name is None


def test_normals_set():
    plane = pyvista.Plane(i_resolution=1, j_resolution=1)
    plane.point_data.normals = plane.point_normals
    assert np.array_equal(plane.point_data.active_normals, plane.point_normals)

    with raises(ValueError, match='must be a 2-dim'):
        plane.point_data.active_normals = [1]
    with raises(ValueError, match='must match number of points'):
        plane.point_data.active_normals = [[1, 1, 1], [0, 0, 0]]
    with raises(ValueError, match='Normals must have exactly 3 components'):
        plane.point_data.active_normals = [[1, 1], [0, 0], [0, 0], [0, 0]]


def test_normals_name(plane):
    plane.clear_data()
    assert plane.point_data.active_normals_name is None

    key = 'data'
    plane.point_data.set_array(plane.point_normals, key)
    plane.point_data.active_normals_name = key
    assert plane.point_data.active_normals_name == key


def test_normals_raise_field(plane):
    with raises(AttributeError):
        plane.field_data.active_normals


def test_add_two_vectors():
    """Ensure we can add two vectors"""
    mesh = pyvista.Plane(i_resolution=1, j_resolution=1)
    mesh.point_data.set_array(range(4), 'my-scalars')
    mesh.point_data.set_array(range(5, 9), 'my-other-scalars')
    vectors0 = np.random.random((4, 3))
    mesh.point_data.set_vectors(vectors0, 'vectors0')
    vectors1 = np.random.random((4, 3))
    mesh.point_data.set_vectors(vectors1, 'vectors1')

    assert 'vectors0' in mesh.point_data
    assert 'vectors1' in mesh.point_data


def test_active_vectors_name_setter():
    mesh = pyvista.Plane(i_resolution=1, j_resolution=1)
    mesh.point_data.set_array(range(4), 'my-scalars')
    vectors0 = np.random.random((4, 3))
    mesh.point_data.set_vectors(vectors0, 'vectors0')
    vectors1 = np.random.random((4, 3))
    mesh.point_data.set_vectors(vectors1, 'vectors1')

    assert mesh.point_data.active_vectors_name == 'vectors1'
    mesh.point_data.active_vectors_name = 'vectors0'
    assert mesh.point_data.active_vectors_name == 'vectors0'

    with raises(KeyError, match='does not contain'):
        mesh.point_data.active_vectors_name = 'not a valid key'

    with raises(ValueError, match='needs 3 components'):
        mesh.point_data.active_vectors_name = 'my-scalars'


def test_active_vectors_eq():
    mesh = pyvista.Plane(i_resolution=1, j_resolution=1)
    vectors0 = np.random.random((4, 3))
    mesh.point_data.set_vectors(vectors0, 'vectors0')
    vectors1 = np.random.random((4, 3))
    mesh.point_data.set_vectors(vectors1, 'vectors1')

    other_mesh = mesh.copy(deep=True)
    assert mesh == other_mesh

    mesh.point_data.active_vectors_name = 'vectors0'
    assert mesh != other_mesh


def test_active_t_coords_name(plane):
    plane.point_data['arr'] = plane.point_data.active_t_coords
    plane.point_data.active_t_coords_name = 'arr'

    with raises(AttributeError):
        plane.field_data.active_t_coords_name = 'arr'


@skip_windows  # windows doesn't support np.complex256
@skip_apple_silicon  # same with Apple silicon (M1/M2)
def test_complex_raises(plane):
    with raises(ValueError, match='Only numpy.complex64'):
        plane.point_data['data'] = np.empty(plane.n_points, dtype=np.complex256)


@mark.parametrize('dtype_str', ['complex64', 'complex128'])
def test_complex(plane, dtype_str):
    """Test if complex data can be properly represented in datasetattributes."""
    dtype = np.dtype(dtype_str)
    name = 'my_data'

    with raises(ValueError, match='Complex data must be single dimensional'):
        plane.point_data[name] = np.empty((plane.n_points, 2), dtype=dtype)

    real_type = np.float32 if dtype == np.complex64 else np.float64
    data = np.random.random((plane.n_points, 2)).astype(real_type).view(dtype).ravel()
    plane.point_data[name] = data
    assert np.array_equal(plane.point_data[name], data)

    assert dtype_str in str(plane.point_data)

    # test setter
    plane.active_scalars_name = name

    # ensure that association is removed when changing datatype
    assert plane.point_data[name].dtype == dtype
    plane.point_data[name] = plane.point_data[name].real
    assert np.issubdtype(plane.point_data[name].dtype, real_type)
