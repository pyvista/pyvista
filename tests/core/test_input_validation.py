import sys
from collections import namedtuple
from re import escape
from typing import Union, get_args, get_origin

import numpy as np
import pytest
from vtk import vtkTransform

from pyvista.core import pyvista_ndarray
from pyvista.core.input_validation.input_validation import (
    _set_default_kwarg_mandatory,
    cast_array_to_NDArray,
    cast_array_to_nested_list,
    cast_array_to_nested_tuple,
    check_has_shape,
    check_is_ArrayLike,
    check_is_DTypeLike,
    check_is_finite,
    check_is_greater_than,
    check_is_in_range,
    check_is_instance,
    check_is_integer,
    check_is_less_than,
    check_is_NDArray,
    check_is_real,
    check_is_sequence,
    check_is_sorted,
    check_is_string,
    check_is_string_sequence,
    check_is_subdtype,
    check_is_type,
    check_sequence_elements_have_type,
    check_string_is_in_list,
    coerce_array_to_arrayNx3,
    coerce_dtypelike_as_dtype,
    coerce_number_or_array3_as_array3,
    coerce_shapelike_as_shape,
    coerce_transformlike_as_array3x3,
    coerce_transformlike_as_array4x4,
    validate_arrayNx3,
    validate_data_range,
    validate_number,
    validate_numeric_array,
)
from pyvista.core.utilities.arrays import vtkmatrix_from_array


@pytest.fixture
def not_ArrayLike():
    return [[1], [2, 3]]


@pytest.mark.parametrize(
    'transform_like',
    [
        np.eye(3),
        np.eye(4),
        np.array(np.eye(3)),
        np.array(np.eye(4)),
        vtkmatrix_from_array(np.eye(3)),
        vtkmatrix_from_array(np.eye(4)),
        vtkTransform(),
    ],
)
def test_coerce_transformlike_as_array4x4(transform_like):
    result = coerce_transformlike_as_array4x4(transform_like)
    assert type(result) is np.ndarray
    assert np.array_equal(result, np.eye(4))


def test_coerce_transformlike_as_array4x4_raises():
    with pytest.raises(ValueError, match=escape("Shape must be one of [(3, 3), (4, 4)]")):
        coerce_transformlike_as_array4x4(np.array([1, 2, 3]))
    with pytest.raises(TypeError, match="must be numeric"):
        coerce_transformlike_as_array4x4("abc")


def test_check_is_subdtype():
    check_is_subdtype(np.array([1, 2, 3]), np.integer)
    check_is_subdtype(np.array([1.0, 2, 3]), dtype=float)
    check_is_subdtype(np.array([1.0, 2, 3], dtype='uint8'), 'uint8')
    check_is_subdtype(np.array([1.0, 2, 3]), ('uint8', float))
    msg = "Input has incorrect dtype of 'int32'. The dtype must be a subtype of <class 'float'>."
    with pytest.raises(TypeError, match=msg):
        check_is_subdtype(np.array([1, 2, 3]), float)
    msg = "Input has incorrect dtype of 'complex128'. The dtype must be a subtype of at least one of \n(<class 'numpy.integer'>, <class 'numpy.floating'>)."
    with pytest.raises(TypeError, match=escape(msg)):
        check_is_subdtype(np.array([1 + 1j, 2, 3]), (np.integer, np.floating))


def test_coerce_dtypelike_as_dtype():
    dtype = coerce_dtypelike_as_dtype('single')
    assert isinstance(dtype, np.dtype)
    assert dtype.type is np.float32

    with pytest.raises(TypeError):
        coerce_dtypelike_as_dtype('cat')


def test_coerce_dtypelike_changes_type():
    # test coercing some types (e.g. np.number) can lead to unexpected
    # failed `np.issubtype` checks due to an implicit change of type
    int_array = np.array([1, 2, 3])
    dtype_expected = np.number
    check_is_subdtype(int_array, dtype_expected)  # int is subtype of np.number

    dtype_coerced = coerce_dtypelike_as_dtype(dtype_expected)
    assert dtype_coerced.type is np.float64  # np.number is coerced (by NumPy) as a float
    with pytest.raises(TypeError):
        # this check will now fail since int is not subtype of float
        check_is_subdtype(int_array, dtype_coerced)


def test_check_is_DTypeLike():
    check_is_DTypeLike(np.number)
    with pytest.raises(TypeError):
        coerce_dtypelike_as_dtype('cat')


#
# from pyvista.core._typing_core import _Sequence3
# def test_check_Sequence_typing():
#     _check_Sequence_typing((1, 2, 3), _Sequence3)

# def test_coerce_ArrayLike_as_NDArray():
#     arr = coerce_ArrayLike_as_NDArray([1,2,3])
#     assert np.array_equal(arr, [1,2,3])
#     with pytest.raises(ValueError, match=escape("Unable to ceorce input as an NDArray. Input must be ArrayLike, got the following instead:\n[[1, 2, 3], [4, 5]]")
# ):
#         arr = coerce_ArrayLike_as_NDArray([[1,2,3],[4,5]])


def test_validate_number():
    num = validate_number(1)
    assert num == 1
    assert type(num) is int

    num = validate_number(2.0, to_list=False, shape=())
    assert num == 2.0
    assert type(num) is np.ndarray
    assert num.dtype.type is np.float64

    msg = "Parameter 'shape' cannot be set for Number. " "Its value is automatically set to `()`."
    with pytest.raises(ValueError, match=escape(msg)):
        validate_number(1, shape=2)


def test_validate_data_range():
    rng = validate_data_range([0, 1])
    assert rng == (0, 1)

    rng = validate_data_range((0, 2.5), to_list=True)
    assert rng == [0.0, 2.5]

    rng = validate_data_range((-10, -10), to_tuple=False, shape=2)
    assert type(rng) is np.ndarray

    msg = "Data Range must be sorted."
    with pytest.raises(ValueError, match=msg):
        validate_data_range((1, 0))

    msg = (
        "Parameter 'shape' cannot be set for Data Range. Its value is " "automatically set to `2`."
    )
    with pytest.raises(ValueError, match=msg):
        validate_data_range((0, 1), shape=3)


def test_set_default_kwarg_mandatory():
    default_value = 1
    default_key = 'k'

    # Test parameter unset
    kwargs = dict()
    _set_default_kwarg_mandatory(kwargs, default_key, default_value)
    assert kwargs[default_key] == default_value

    # Test parameter already set to default
    kwargs = dict()
    kwargs[default_key] = default_value
    _set_default_kwarg_mandatory(kwargs, default_key, default_value)
    assert kwargs[default_key] == default_value

    # Test parameter set to non-default
    kwargs = dict()
    kwargs[default_key] = default_value * 2
    msg = "Parameter 'k' cannot be set for Array. Its value is " "automatically set to `1`."
    with pytest.raises(ValueError, match=msg):
        _set_default_kwarg_mandatory(kwargs, default_key, default_value)


def test_check_array_shape():
    check_has_shape(0, ())
    check_has_shape(0, [(), 2])
    check_has_shape((1, 2, 3), [(), 3])
    check_has_shape((1, 2, 3), [-1])
    check_has_shape((1, 2, 3), -1)

    msg = 'Input has shape (3,) which is not allowed. Shape must be 0.'
    with pytest.raises(ValueError, match=escape(msg)):
        check_has_shape((1, 2, 3), 0, name="Input")

    msg = 'Array has shape (3,) which is not allowed. Shape must be one of [(), (4, 5)].'
    with pytest.raises(ValueError, match=escape(msg)):
        check_has_shape((1, 2, 3), [(), (4, 5)])


def test_coerce_shapelike_as_shape():
    msg = "`None` is not a valid shape. Use `()` instead."
    with pytest.raises(TypeError, match=escape(msg)):
        coerce_shapelike_as_shape(None)
    shape = coerce_shapelike_as_shape(())
    assert shape == ()
    shape = coerce_shapelike_as_shape(1)
    assert shape == (1,)
    shape = coerce_shapelike_as_shape(-1)
    assert shape == (-1,)
    shape = coerce_shapelike_as_shape((1, 2, 3))
    assert shape == (
        1,
        2,
        3,
    )
    shape = coerce_shapelike_as_shape((-1, 2, -1))
    assert shape == (-1, 2, -1)

    msg = (
        "Shape has incorrect dtype of 'float64'. "
        "The dtype must be a subtype of <class 'numpy.integer'>."
    )
    with pytest.raises(TypeError, match=escape(msg)):
        coerce_shapelike_as_shape(1.0)

    msg = "Shape values must all be greater than or equal to -1."
    with pytest.raises(ValueError, match=msg):
        coerce_shapelike_as_shape(-2)

    msg = "Shape must be scalar or 1-dimensional."
    with pytest.raises(ValueError, match=msg):
        coerce_shapelike_as_shape(((1, 2), (3, 4)))


def test_coerce_array_to_shapeNx3():
    arr = coerce_array_to_arrayNx3((1, 2, 3))
    assert arr.shape == (1, 3)
    assert np.array_equal(arr, [[1, 2, 3]])

    arr = coerce_array_to_arrayNx3([(1, 2, 3), (4, 5, 6)])
    assert arr.shape == (2, 3)

    msg = (
        "Parameter 'shape' cannot be set for Array. Its value is "
        "automatically set to `[3, (-1, 3)]`."
    )
    with pytest.raises(ValueError, match=escape(msg)):
        coerce_array_to_arrayNx3((1, 2, 3), shape=1)
    msg = "Array has shape () which is not allowed. Shape must be one of [3, (-1, 3)]."
    with pytest.raises(ValueError, match=escape(msg)):
        coerce_array_to_arrayNx3(0)
    msg = "Array has shape (4,)"
    with pytest.raises(ValueError, match=escape(msg)):
        coerce_array_to_arrayNx3([1, 2, 3, 4])


def test_check_is_in_range():
    check_is_in_range((1, 2, 3), [1, 3])

    msg = "Array values must all be less than or equal to 2."
    with pytest.raises(ValueError, match=msg):
        check_is_in_range((1, 2, 3), [1, 2])

    msg = "Input values must all be greater than or equal to 2."
    with pytest.raises(ValueError, match=msg):
        check_is_in_range((1, 2, 3), [2, 3], name='Input')

    # Test strict bounds
    msg = "Array values must all be less than 3."
    with pytest.raises(ValueError, match=msg):
        check_is_in_range((1, 2, 3), [1, 3], strict_upper=True)

    msg = "Array values must all be greater than 1."
    with pytest.raises(ValueError, match=msg):
        check_is_in_range((1, 2, 3), [1, 3], strict_lower=True)


# def test_check_string_is_in_list():
#     check_is_string("abc")
#     check_is_string("abc", name='123')
#     msg = "Value must be a string, got <class 'int'> instead."
#     with pytest.raises(TypeError, match=msg):
#         check_is_string(0, name='Value')
#     msg = "Input must be a string, got <class 'int'> instead."
#     with pytest.raises(TypeError, match=msg):
#         check_is_string(0)
#     msg = "Name must be a string, got <class 'float'> instead."
#     with pytest.raises(TypeError, match=msg):
#         check_is_string("abc", name=0.0)


def numeric_array_test_cases():
    Case = namedtuple("Case", ["kwarg", "valid_array", "invalid_array", "error_type", "error_msg"])
    return (
        Case(dict(must_be_finite=True), 0, np.inf, ValueError, 'must have finite values'),
        Case(dict(must_be_real=True), 0, 1 + 1j, TypeError, 'must have real numbers'),
        Case(
            dict(must_be_integer_like=True), 0.0, 0.1, ValueError, 'must have integer-like values'
        ),
        Case(dict(must_be_sorted=True), [0, 1], [1, 0], ValueError, 'must be sorted'),
    )


@pytest.mark.parametrize('name', ["_array", "_input"])
@pytest.mark.parametrize('copy', [True, False])
@pytest.mark.parametrize('as_any', [True, False])
@pytest.mark.parametrize('to_list', [True, False])
@pytest.mark.parametrize('to_tuple', [True, False])
@pytest.mark.parametrize('dtype_out', [np.float32, np.float64])
@pytest.mark.parametrize('case', numeric_array_test_cases())
@pytest.mark.parametrize('stack_input', [True, False])
@pytest.mark.parametrize('input_type', [tuple, list, np.ndarray, pyvista_ndarray])
def test_validate_numeric_array_cases(
    name, copy, as_any, to_list, to_tuple, dtype_out, case, stack_input, input_type
):
    # Set up
    valid_array = np.array(case.valid_array)
    invalid_array = np.array(case.invalid_array)

    # Inputs may be scalar, use stacking to ensure we have test cases
    # with multidimensional arrays
    if stack_input:
        valid_array = np.stack((valid_array, valid_array), axis=0)
        valid_array = np.stack((valid_array, valid_array), axis=1)
        invalid_array = np.stack((invalid_array, invalid_array), axis=0)
        invalid_array = np.stack((invalid_array, invalid_array), axis=1)

    if input_type is tuple:
        valid_array = cast_array_to_nested_tuple(valid_array)
        invalid_array = cast_array_to_nested_tuple(invalid_array)
    elif input_type is list:
        valid_array = valid_array.tolist()
        invalid_array = invalid_array.tolist()
    elif input_type is np.ndarray:
        valid_array = np.asarray(valid_array)
        invalid_array = np.asarray(invalid_array)
    else:  # pyvista_ndarray:
        valid_array = pyvista_ndarray(valid_array)
        invalid_array = pyvista_ndarray(invalid_array)

    common_kwargs = dict(
        **case.kwarg,
        name=name,
        copy=copy,
        as_any=as_any,
        to_list=to_list,
        to_tuple=to_tuple,
        dtype_out=dtype_out,
    )

    # Test raises correct error with invalid input
    with pytest.raises(case.error_type, match=case.error_msg):
        validate_numeric_array(invalid_array, **common_kwargs)
    # Test error has correct name
    with pytest.raises(case.error_type, match=name):
        validate_numeric_array(invalid_array, **common_kwargs)

    # Test no error with valid input
    array_in = valid_array
    array_out = validate_numeric_array(array_in, **common_kwargs)
    assert np.array_equal(array_out, array_in)

    # Check output

    if np.array(array_in).ndim == 0 and (to_tuple or to_list):
        # test scalar input results in scalar output
        assert type(array_out) is float or type(array_out) is int
    elif to_tuple:
        assert type(array_out) is tuple
    elif to_list:
        assert type(array_out) is list
    else:
        assert isinstance(array_out, np.ndarray)
        assert array_out.dtype.type is dtype_out
        if as_any:
            if input_type is pyvista_ndarray:
                assert type(array_out) is pyvista_ndarray
            elif input_type is np.ndarray:
                assert type(array_out) is np.ndarray
            if (
                not copy
                and isinstance(array_in, np.ndarray)
                and coerce_dtypelike_as_dtype(dtype_out) is array_in.dtype
            ):
                assert array_out is array_in
            else:
                assert array_out is not array_in
        else:
            assert type(array_out) is np.ndarray

    if copy:
        assert array_out is not array_in


@pytest.mark.parametrize('object', [0, 0.0, "0"])
@pytest.mark.parametrize(
    'classinfo', [int, (int, float), [int, float]]
)
@pytest.mark.parametrize('allow_subclass', [True, False])
# @pytest.mark.parametrize('name', [True,False])
def test_check_is_instance(object, classinfo, allow_subclass):
    if isinstance(classinfo, list):
        with pytest.raises(TypeError):
            check_is_instance(object, classinfo)
        return

    if allow_subclass:
        if isinstance(object, classinfo):
            check_is_instance(object, classinfo)
        else:
            with pytest.raises(TypeError, match='Object must be an instance of'):
                check_is_instance(object, classinfo)
            with pytest.raises(TypeError, match='Input must be an instance of'):
                check_is_instance(object, classinfo, name='Input')

    else:
        if type(classinfo) is tuple:
            if type(object) in classinfo:
                check_is_type(object, classinfo)
            else:
                with pytest.raises(TypeError, match='Input must have one of the following types'):
                    check_is_type(object, classinfo, name='Input')
                with pytest.raises(TypeError, match='Object must have one of the following types'):
                    check_is_type(object, classinfo)
        elif get_origin(classinfo) is Union:
            if type(object) in get_args(classinfo):
                check_is_type(object, classinfo)
            else:
                with pytest.raises(TypeError, match='Input must have one of the following types'):
                    check_is_type(object, classinfo, name='Input')
                with pytest.raises(TypeError, match='Object must have one of the following types'):
                    check_is_type(object, classinfo)
        else:
            if type(object) is classinfo:
                check_is_type(object, classinfo)
            else:
                with pytest.raises(TypeError, match='Input must have type'):
                    check_is_type(object, classinfo, name='Input')
                with pytest.raises(TypeError, match='Object must have type'):
                    check_is_type(object, classinfo)

    msg = "Name must be a string, got <class 'int'> instead."
    with pytest.raises(TypeError, match=msg):
        check_is_instance(0, int, name=0)


def test_check_is_type():
    check_is_type(0, int, name='abc')
    check_is_type(0, Union[int])
    with pytest.raises(TypeError):
        check_is_type("str", int)
    with pytest.raises(TypeError):
        check_is_type(0, int, name=1)

    if sys.version_info < (3, 10):
        msg = "Subscripted generics cannot be used with class and instance checks"
        with pytest.raises(TypeError, match=msg):
            check_is_type(0, Union[int, float])
    else:
        check_is_type(0, Union[int, float])


def test_check_is_string():
    check_is_string("abc")
    check_is_string("abc", name='123')
    msg = "Value must be an instance of <class 'str'>. Got <class 'int'> instead."
    with pytest.raises(TypeError, match=msg):
        check_is_string(0, name='Value')
    msg = "Object must be an instance of <class 'str'>. Got <class 'int'> instead."
    with pytest.raises(TypeError, match=msg):
        check_is_string(0)
    msg = "Name must be an instance of <class 'str'>. Got <class 'float'> instead."
    with pytest.raises(TypeError, match=msg):
        check_is_string("abc", name=0.0)

    class str_subclass(str):
        pass

    check_is_string(str_subclass(), allow_subclass=True)
    with pytest.raises(TypeError, match="Object must have type <class 'str'>."):
        check_is_string(str_subclass(), allow_subclass=False)


def test_check_is_ArrayLike():
    check_is_ArrayLike([1, 2])
    with pytest.raises(ValueError, match="_input"):
        check_is_ArrayLike([[1], [2, 3]], name="_input")


@pytest.mark.parametrize('as_any', [True, False])
@pytest.mark.parametrize('copy', [True, False])
@pytest.mark.parametrize('dtype', [None, float])
def test_cast_as_NDArray(as_any, copy, dtype):
    array_in = pyvista_ndarray([1, 2])
    array_out = cast_array_to_NDArray(array_in, copy=copy, as_any=as_any, dtype=dtype)
    assert np.array_equal(array_out, array_in)
    if as_any:
        assert type(array_out) is pyvista_ndarray
    else:
        assert type(array_out) is np.ndarray

    if copy:
        assert array_out is not array_in

    if dtype is None:
        assert array_out.dtype.type is array_in.dtype.type
    else:
        assert array_out.dtype.type is np.dtype(dtype).type


def test_cast_to_NDArray_raises():
    msg = "Input cannot be cast as <class 'numpy.ndarray'>."
    with pytest.raises(ValueError, match=msg):
        cast_array_to_NDArray([[1], [2, 3]])


def test_cast_to_tuple_array(not_ArrayLike):
    array_in = np.zeros(shape=(2, 2, 3))
    array_tuple = cast_array_to_nested_tuple(array_in)
    assert array_tuple == (((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    array_list = array_in.tolist()
    assert np.array_equal(array_tuple, array_list)
    with pytest.raises(ValueError, match="_input"):
        cast_array_to_nested_tuple(not_ArrayLike, name='_input')


def test_cast_to_list_array(not_ArrayLike):
    array_in = np.zeros(shape=(3, 4, 5))
    array_list = cast_array_to_nested_list(array_in)
    assert np.array_equal(array_in, array_list)
    with pytest.raises(ValueError, match="_input"):
        cast_array_to_nested_list(not_ArrayLike, name='_input')


def test_coerce_transformlike_as_array3x3():
    coerce_transformlike_as_array3x3


def test_check_sequence_elements_have_type():
    check_sequence_elements_have_type


def test_validate_arrayNx3():
    validate_arrayNx3


def test_check_is_less_than():
    check_is_less_than


def test_check_is_greater_than():
    check_is_greater_than


def test_check_is_real():
    check_is_real


def test_check_is_finite():
    check_is_finite


def test_check_is_integer():
    check_is_integer


def test_check_is_sequence():
    check_is_sequence


def test_check_is_sorted():
    check_is_sorted


def test_check_is_string_sequence():
    check_is_string_sequence


def test_check_is_NDArray():
    check_is_NDArray


def test_coerce_number_or_array3_as_array3():
    coerce_number_or_array3_as_array3


def test_check_string_is_in_list():
    check_string_is_in_list
