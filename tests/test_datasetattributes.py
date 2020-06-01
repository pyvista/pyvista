import sys
from string import ascii_letters, digits, whitespace

import numpy as np
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import integers, lists, text
from pytest import fixture, mark, raises

import pyvista
from pyvista.utilities import FieldAssociation


@fixture()
def hexbeam_point_attributes(hexbeam):
    return hexbeam.point_arrays


@fixture()
def hexbeam_field_attributes(hexbeam):
    return hexbeam.field_arrays


@fixture()
def insert_arange_narray(hexbeam_point_attributes):
    n_points = hexbeam_point_attributes.dataset.GetNumberOfPoints()
    sample_array = np.arange(n_points)
    hexbeam_point_attributes.append(sample_array, 'sample_array')
    return hexbeam_point_attributes, sample_array


@fixture()
def insert_bool_array(hexbeam_point_attributes):
    n_points = hexbeam_point_attributes.dataset.GetNumberOfPoints()
    sample_array = np.ones(n_points, np.bool_)
    hexbeam_point_attributes.append(sample_array, 'sample_array')
    return hexbeam_point_attributes, sample_array


def test_repr(hexbeam_point_attributes):
    assert 'POINT' in str(hexbeam_point_attributes)
    assert 'DataSetAttributes' in str(hexbeam_point_attributes)
    assert 'Contains keys:' in str(hexbeam_point_attributes)


def test_init(hexbeam):
    attributes = pyvista.DataSetAttributes(
        hexbeam.GetPointData(), dataset=hexbeam, association=FieldAssociation.POINT)
    assert attributes.VTKObject == hexbeam.GetPointData()
    assert attributes.dataset == hexbeam
    assert attributes.association == FieldAssociation.POINT


def test_empty_active_vectors(hexbeam):
    assert hexbeam.active_vectors is None


def test_valid_array_len_cells(hexbeam):
    assert hexbeam.cell_arrays.valid_array_len == hexbeam.n_cells


def test_append_matrix(hexbeam):
    mat_shape = (hexbeam.n_points, 3, 2)
    mat = np.random.random(mat_shape)
    hexbeam.point_arrays.append(mat, 'mat')
    matout = hexbeam.point_arrays['mat'].reshape(mat_shape)
    assert np.allclose(mat, matout)


def test_set_vectors(hexbeam):
    vectors = np.random.random((hexbeam.n_points, 3))
    hexbeam['vectors'] = vectors
    hexbeam.set_active_vectors('vectors')
    assert np.allclose(hexbeam.active_vectors, vectors)


def test_set_active_vectors_invalid(hexbeam):
    # verify non-vector data does not become active vectors
    not_vectors = np.random.random((hexbeam.points.shape))
    hexbeam.point_arrays['not_vectors'] = not_vectors
    assert np.allclose(hexbeam.point_arrays.active_vectors, not_vectors)


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


def test_append_should_not_add_none_array(hexbeam_point_attributes):
    with raises(TypeError):
        hexbeam_point_attributes.append(None, 'sample_array')


def test_append_should_contain_array_name(insert_arange_narray):
    dsa, _ = insert_arange_narray
    assert 'sample_array' in dsa


def test_append_should_contain_exact_array(insert_arange_narray):
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


@settings(max_examples=20)
@given(scalar=integers(min_value=-sys.maxsize - 1, max_value=sys.maxsize))
def test_append_should_accept_scalar_value(scalar, hexbeam_point_attributes):
    hexbeam_point_attributes.append(narray=scalar, name='int_array')


@settings(max_examples=20)
@given(scalar=integers(min_value=-sys.maxsize - 1, max_value=sys.maxsize))
def test_append_scalar_value_should_give_array(scalar, hexbeam_point_attributes):
    hexbeam_point_attributes.append(narray=scalar, name='int_array')
    expected = np.full(hexbeam_point_attributes.dataset.n_points, scalar)
    assert np.array_equal(expected, hexbeam_point_attributes['int_array'])


@given(arr=lists(text(alphabet=ascii_letters + digits + whitespace), max_size=16))
def test_append_string_lists_should_equal(arr, hexbeam_field_attributes):
    hexbeam_field_attributes['string_arr'] = arr
    assert arr == hexbeam_field_attributes['string_arr'].tolist()


@given(arr=arrays(dtype='U', shape=10))
def test_append_string_array_should_equal(arr, hexbeam_field_attributes):
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


def test_pop_should_return_array(insert_arange_narray):
    dsa, sample_array = insert_arange_narray
    other_array = dsa.pop('sample_array')
    assert np.array_equal(other_array, sample_array)


@mark.parametrize('removed_key', [None, 'nonexistant_array_name', '', -1])
def test_remove_should_fail_on_bad_argument(removed_key, hexbeam_point_attributes):
    with raises(KeyError):
        hexbeam_point_attributes.remove(removed_key)


@mark.parametrize('removed_key', [None, 'nonexistant_array_name', '', -1])
def test_del_should_fail_bad_argument(removed_key, hexbeam_point_attributes):
    with raises(KeyError):
        del hexbeam_point_attributes[removed_key]


@mark.parametrize('removed_key', [None, 'nonexistant_array_name', '', -1])
def test_pop_should_fail_bad_argument(removed_key, hexbeam_point_attributes):
    with raises(KeyError):
        hexbeam_point_attributes.pop(removed_key)


def test_length_should_increment_on_append(hexbeam_point_attributes):
    initial_len = len(hexbeam_point_attributes)
    n_points = hexbeam_point_attributes.dataset.GetNumberOfPoints()
    sample_array = np.arange(n_points)
    hexbeam_point_attributes.append(sample_array, 'sample_array')
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
        assert(type(name) == str)


def test_key_should_exist(insert_arange_narray):
    dsa, sample_array = insert_arange_narray
    assert 'sample_array' in dsa.keys()


def test_values_should_be_pyvista_ndarrays(insert_arange_narray):
    dsa, sample_array = insert_arange_narray
    for arr in dsa.values():
        assert(type(arr) == pyvista.pyvista_ndarray)


def test_value_should_exist(insert_arange_narray):
    dsa, sample_array = insert_arange_narray
    for arr in dsa.values():
        if np.array_equal(sample_array, arr):
            return
    raise AssertionError('Array not in values.')


def test_active_scalars_setter(hexbeam_point_attributes):
    dsa = hexbeam_point_attributes
    assert dsa.active_scalars is None

    dsa.active_scalars = 'sample_point_scalars'
    assert dsa.active_scalars is not None
    assert dsa.GetScalars().GetName() == 'sample_point_scalars'
