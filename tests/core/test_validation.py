from collections import namedtuple
from copy import deepcopy
from enum import Enum, auto
import inspect
import itertools
from numbers import Real
from re import escape
import sys
from typing import Any, Callable, Dict, Union, get_args, get_origin

import numpy as np
import pytest
from vtk import vtkTransform

from pyvista.core import pyvista_ndarray
from pyvista.core._validation import (
    check_contains,
    check_finite,
    check_greater_than,
    check_instance,
    check_integer,
    check_iterable,
    check_iterable_items,
    check_length,
    check_less_than,
    check_ndim,
    check_nonnegative,
    check_number,
    check_range,
    check_real,
    check_sequence,
    check_shape,
    check_sorted,
    check_string,
    check_subdtype,
    check_type,
    validate_array,
    validate_array3,
    validate_arrayN,
    validate_arrayN_unsigned,
    validate_arrayNx3,
    validate_axes,
    validate_data_range,
    validate_number,
    validate_transform3x3,
    validate_transform4x4,
)
from pyvista.core._validation._array_wrapper import (
    _ArrayLikeWrapper,
    _NestedSequenceWrapper,
    _NumberWrapper,
    _NumpyArrayWrapper,
    _SequenceWrapper,
)
from pyvista.core._validation._cast_array import _cast_to_list, _cast_to_numpy, _cast_to_tuple
from pyvista.core._validation.check import _validate_shape_value
from pyvista.core._validation.validate import _array_from_vtkmatrix
from pyvista.core._vtk_core import vtkMatrix3x3, vtkMatrix4x4
from pyvista.core.utilities.arrays import array_from_vtkmatrix, vtkmatrix_from_array


@pytest.mark.parametrize(
    'transform_like',
    [
        np.eye(3),
        np.eye(4),
        np.eye(3).tolist(),
        np.eye(4).tolist(),
        vtkmatrix_from_array(np.eye(3)),
        vtkmatrix_from_array(np.eye(4)),
        vtkTransform(),
    ],
)
def test_validate_transform4x4(transform_like):
    result = validate_transform4x4(transform_like)
    assert type(result) is np.ndarray
    assert np.array_equal(result, np.eye(4))


def test_validate_transform4x4_raises():
    with pytest.raises(TypeError, match=escape("Input transform must be one of")):
        validate_transform4x4(np.array([1, 2, 3]))
    with pytest.raises(TypeError, match="must have real numbers"):
        validate_transform4x4("abc")


@pytest.mark.parametrize(
    'transform_like',
    [
        np.eye(3),
        np.eye(3).tolist(),
        vtkmatrix_from_array(np.eye(3)),
    ],
)
def test_validate_transform3x3(transform_like):
    result = validate_transform3x3(transform_like)
    assert type(result) is np.ndarray
    assert np.array_equal(result, np.eye(3))


def test_validate_transform3x3_raises():
    with pytest.raises(TypeError, match=escape("Input transform must be one of")):
        validate_transform3x3(np.array([1, 2, 3]))
    with pytest.raises(TypeError, match="must have real numbers."):
        validate_transform3x3("abc")


def test_check_subdtype():
    check_subdtype(int, np.integer)
    check_subdtype(np.dtype(int), np.integer)
    check_subdtype(np.array([1, 2, 3]), np.integer)
    check_subdtype([1.0, 2, 3], float)
    check_subdtype(np.array([1.0, 2, 3], dtype='uint8'), 'uint8')
    check_subdtype(np.array([1.0, 2, 3]), ('uint8', float))
    msg = "Input has incorrect dtype of dtype('int32'). The dtype must be a subtype of <class 'float'>."
    with pytest.raises(TypeError, match=escape(msg)):
        check_subdtype(np.array([1, 2, 3]).astype('int32'), float)
    msg = "Input has incorrect dtype of dtype('complex128'). The dtype must be a subtype of at least one of \n(<class 'numpy.integer'>, <class 'numpy.floating'>)."
    with pytest.raises(TypeError, match=escape(msg)):
        check_subdtype(np.array([1 + 1j, 2, 3]), (np.integer, np.floating))


def test_check_subdtype_changes_type():
    # test coercing some types (e.g. np.number) can lead to unexpected
    # failed `np.issubtype` checks due to an implicit change of type
    int_array = np.array([1, 2, 3])
    dtype_expected = np.number
    check_subdtype(int_array, dtype_expected)  # int is subtype of np.number

    dtype_coerced = np.dtype(dtype_expected)
    assert dtype_coerced.type is np.float64  # np.number is coerced (by NumPy) as a float
    with pytest.raises(TypeError):
        # this check will now fail since int is not subtype of float
        check_subdtype(int_array, dtype_coerced)


def test_validate_number():
    num, flags = validate_number(
        [2.0],
        reshape=True,
        must_be_finite=True,
        must_be_real=True,
        must_have_dtype=None,
        must_be_nonnegative=True,
        must_be_integer=True,
        must_be_in_range=[0, 3],
        strict_lower_bound=True,
        strict_upper_bound=True,
        dtype_out=int,
        get_flags=True,
        name='_number',
    )
    assert num == 2
    assert isinstance(num, int)

    num = validate_number(np.array([3.0]))
    assert num == 3.0
    assert isinstance(num, float)

    msg = 'Number has shape (1,) which is not allowed. Shape must be ().'
    with pytest.raises(ValueError, match=escape(msg)):
        validate_number([1], reshape=False)


def test_validate_data_range():
    rng = validate_data_range([0, 1])
    assert rng == (0, 1)

    rng = validate_data_range((0, 2.5))
    assert rng == (0, 2.5)
    assert isinstance(rng[0], int)
    assert isinstance(rng[1], float)

    rng = validate_data_range((0, 2.5), dtype_out=float)
    assert rng == (0.0, 2.5)

    msg = 'Data Range with 2 elements must be sorted in ascending order. Got:\n    (1, 0)'
    with pytest.raises(ValueError, match=escape(msg)):
        validate_data_range((1, 0))


def test_check_shape():
    check_shape(0, ())
    check_shape(0, [(), 2])
    check_shape((1, 2, 3), [(), 3])
    check_shape((1, 2, 3), [-1])
    check_shape((1, 2, 3), -1)

    msg = 'Input has shape (3,) which is not allowed. Shape must be 0.'
    with pytest.raises(ValueError, match=escape(msg)):
        check_shape((1, 2, 3), 0, name="Input")

    msg = 'Array has shape (3,) which is not allowed. Shape must be one of [(), (4, 5)].'
    with pytest.raises(ValueError, match=escape(msg)):
        check_shape((1, 2, 3), [(), (4, 5)])


def test_check_ndim():
    check_ndim(0, 0)
    check_ndim(np.array(0), 0)
    check_ndim((1, 2, 3), range(2))
    check_ndim([[1, 2, 3]], (0, 2))

    msg = 'Input has the incorrect number of dimensions. Got 1, expected 0.'
    with pytest.raises(ValueError, match=escape(msg)):
        check_ndim((1, 2, 3), 0, name="Input")

    msg = 'Array has the incorrect number of dimensions. Got 1, expected one of [4, 5].'
    with pytest.raises(ValueError, match=escape(msg)):
        check_ndim((1, 2, 3), [4, 5])


def test_validate_shape_value():
    msg = "`None` is not a valid shape. Use `()` instead."
    with pytest.raises(TypeError, match=escape(msg)):
        _validate_shape_value(None)
    shape = _validate_shape_value(())
    assert shape == ()
    shape = _validate_shape_value(1)
    assert shape == (1,)
    shape = _validate_shape_value(-1)
    assert shape == (-1,)
    shape = _validate_shape_value((1, 2, 3))
    assert shape == (
        1,
        2,
        3,
    )
    shape = _validate_shape_value((-1, 2, -1))
    assert shape == (-1, 2, -1)

    msg = (
        "Shape must be an instance of any type (<class 'int'>, <class 'tuple'>). "
        "Got <class 'float'> instead."
    )
    with pytest.raises(TypeError, match=escape(msg)):
        _validate_shape_value(1.0)

    msg = "Shape values must all be greater than or equal to -1."
    with pytest.raises(ValueError, match=msg):
        _validate_shape_value(-2)

    msg = "All items of Shape must be an instance of <class 'int'>. Got <class 'tuple'> instead."
    with pytest.raises(TypeError, match=msg):
        _validate_shape_value(((1, 2), (3, 4)))


@pytest.mark.parametrize('reshape', [True, False])
def test_validate_arrayNx3(reshape):
    arr = validate_arrayNx3((1, 2, 3))
    assert np.shape(arr) == (1, 3)
    assert np.array_equal(arr, [[1, 2, 3]])

    if not reshape:
        msg = "Array has shape (3,) which is not allowed. Shape must be (-1, 3)."
        with pytest.raises(ValueError, match=escape(msg)):
            validate_arrayNx3((1, 2, 3), reshape=False)

    arr = validate_arrayNx3([(1, 2, 3), (4, 5, 6)], reshape=reshape)
    assert np.shape(arr) == (2, 3)

    msg = "Array has shape () which is not allowed. Shape must be one of [(-1, 3), (3,)]."
    with pytest.raises(ValueError, match=escape(msg)):
        validate_arrayNx3(0)
    with pytest.raises(ValueError, match="_input"):
        validate_arrayNx3([1, 2, 3, 4], name="_input")


@pytest.mark.parametrize('reshape', [True, False])
def test_validate_arrayN(reshape):
    # test 0D input is reshaped to 1D by default
    arr = validate_arrayN(0)
    assert np.shape(arr) == (1,)
    assert np.array_equal(arr, [0])

    # test 2D input is reshaped to 1D by default
    arr = validate_arrayN([[1, 2, 3]])
    assert np.shape(arr) == (3,)
    assert np.array_equal(arr, [1, 2, 3])

    arr = validate_arrayN([[1], [2], [3]])
    assert np.shape(arr) == (3,)
    assert np.array_equal(arr, [1, 2, 3])

    if not reshape:
        msg = 'Array has shape () which is not allowed. Shape must be (-1,).'
        with pytest.raises(ValueError, match=escape(msg)):
            validate_arrayN(0, reshape=False)

        msg = 'Array has shape (1, 3) which is not allowed. Shape must be (-1,).'
        with pytest.raises(ValueError, match=escape(msg)):
            validate_arrayN([[1, 2, 3]], reshape=False)

    arr = validate_arrayN((1, 2, 3, 4, 5, 6), reshape=reshape)
    assert np.shape(arr) == (6,)

    msg = 'Array has shape (2, 2) which is not allowed. Shape must be one of [(), -1, (1, -1), (-1, 1)].'
    with pytest.raises(ValueError, match=escape(msg)):
        validate_arrayN(((1, 2), (3, 4)))
    with pytest.raises(ValueError, match="_input"):
        validate_arrayN(((1, 2), (3, 4)), name="_input")


@pytest.mark.parametrize('reshape', [True, False])
def test_validate_arrayN_unsigned(reshape):
    # test 0D input is reshaped to 1D by default
    arr = validate_arrayN_unsigned(0.0)
    assert np.shape(arr) == (1,)
    assert np.array_equal(arr, [0])
    assert isinstance(arr, np.ndarray)

    arr = validate_arrayN_unsigned(0.0, dtype_out=np.uint8)
    assert arr.dtype.type is np.uint8

    with pytest.raises(ValueError, match=escape('Shape must be (-1,).')):
        validate_arrayN_unsigned(0.0, reshape=False)

    msg = '_input values must all be greater than or equal to 0.'
    with pytest.raises(ValueError, match=msg):
        validate_arrayN_unsigned([-1, 1], name="_input")


@pytest.mark.parametrize('reshape', [True, False])
def test_validate_array3(reshape):
    arr, flags = validate_array3(
        (1, 2, 3),
        reshape=reshape,
        broadcast=True,
        must_have_dtype=int,
        must_be_finite=True,
        must_be_real=True,
        must_be_integer=True,
        must_be_nonnegative=True,
        must_be_sorted=True,
        must_be_in_range=[0, 4],
        strict_lower_bound=True,
        strict_upper_bound=True,
        dtype_out=int,
        as_any=True,
        copy=True,
        get_flags=True,
        name='_array',
    )
    assert np.array_equal(arr, (1, 2, 3))
    assert isinstance(arr, np.ndarray)

    # test 0D input is reshaped to len-3 1D vector with broadcasting enabled
    arr = validate_array3(0, broadcast=True)
    assert np.shape(arr) == (3,)
    assert np.array_equal(arr, [0, 0, 0])

    # test 2D input is reshaped to 1D by default
    arr = validate_array3([[1, 2, 3]])
    assert np.shape(arr) == (3,)
    assert np.array_equal(arr, [1, 2, 3])

    arr = validate_array3([[1], [2], [3]])
    assert np.shape(arr) == (3,)
    assert np.array_equal(arr, [1, 2, 3])

    if not reshape:
        # test check fails with 2D input and no reshape
        msg = 'Array has shape (1, 3) which is not allowed. Shape must be (3,).'
        with pytest.raises(ValueError, match=escape(msg)):
            validate_array3([[1, 2, 3]], reshape=reshape)

        # test correct shape with broadcast and no reshape
        msg = "Shape must be one of [(3,), (), (1,)]."
        with pytest.raises(ValueError, match=escape(msg)):
            validate_array3((1, 2, 3, 4, 5, 6), reshape=reshape, broadcast=True)
    else:
        # test error msg shows correct shape with broadcast and with reshape
        msg = "Shape must be one of [(3,), (1, 3), (3, 1), (), (1,)]"
        with pytest.raises(ValueError, match=escape(msg)):
            validate_array3((1, 2, 3, 4, 5, 6), reshape=reshape, broadcast=True)


def test_check_range():
    check_range((1, 2, 3), [1, 3])

    msg = "Array values must all be less than or equal to 2."
    with pytest.raises(ValueError, match=msg):
        check_range((1, 2, 3), [1, 2])

    msg = "Input values must all be greater than or equal to 2."
    with pytest.raises(ValueError, match=msg):
        check_range((1, 2, 3), [2, 3], name='Input')

    # Test strict bounds
    msg = "Array values must all be less than 3."
    with pytest.raises(ValueError, match=msg):
        check_range((1, 2, 3), [1, 3], strict_upper=True)

    msg = "Array values must all be greater than 1."
    with pytest.raises(ValueError, match=msg):
        check_range((1, 2, 3), [1, 3], strict_lower=True)


def _generate_ids(name, values):
    """Generate named test ids using param name and values."""
    name_iter = [name] * len(values)
    format_id = lambda name, val: f'{name}={val.__name__ if type(val) is type else val}'.replace(
        ' ',
        '',
    )
    return [format_id(name, val) for name, val in zip(name_iter, values)]


def parametrize_with_ids(name, values):
    """Give meaningful names to parametrized tests."""
    return pytest.mark.parametrize(name, values, ids=_generate_ids(name, values))


@parametrize_with_ids('copy', [True, False])
@parametrize_with_ids('as_any', [True, False])
@parametrize_with_ids('dtype_in', [float, int])
@parametrize_with_ids('dtype_out', [float, int, np.float32, np.float64])
@parametrize_with_ids('array', [0, [0, 1], [[0, 1], [0, 1]]])
@parametrize_with_ids('input_type', [tuple, list, np.ndarray, pyvista_ndarray])
@parametrize_with_ids('return_type', [tuple, list, np.ndarray, None])
def test_validate_array(
    copy,
    as_any,
    dtype_in,
    dtype_out,
    array,
    input_type,
    return_type,
):
    # Set up
    def _setup_array_type_and_dtype(array_, input_type_, dtype_in_):
        array_setup = np.array(array_)
        array_setup = array_setup.astype(dtype_in_)

        if input_type_ is tuple:
            return _cast_to_tuple(array_setup)
        elif input_type_ is list:
            return array_setup.tolist()
        elif input_type_ is np.ndarray:
            return np.asarray(array_setup)
        else:  # pyvista_ndarray:
            return pyvista_ndarray(array_setup)

    array_in = _setup_array_type_and_dtype(array, input_type, dtype_in)

    has_numpy_dtype = issubclass(dtype_out, np.generic)
    if has_numpy_dtype and return_type not in [None, np.ndarray, 'numpy']:
        pytest.skip('NumPy dtype specified with non-numpy return type')

    # These are actual parametrized test keywords
    test_kwargs = dict(
        copy=copy,
        as_any=as_any,
        dtype_out=dtype_out,
        return_type=return_type,
    )

    # Also include other keywords dynamically based on input array
    shape = np.array(array_in).shape
    dynamic_kwargs = dict(
        must_have_shape=shape,
        must_have_dtype=np.number,
        must_have_length=range(np.array(array_in).size + 1),
        must_have_min_length=1,
        must_have_max_length=np.array(array_in).size,
        reshape_to=shape,
        broadcast_to=shape,
        must_be_in_range=(np.min(array_in), np.max(array_in)),
        must_be_nonnegative=np.all(np.array(array_in) > 0),
    )
    common_kwargs = {**test_kwargs, **dynamic_kwargs}

    # Do test
    array_out, flags = validate_array(array_in, **common_kwargs, get_flags=True)
    assert np.array_equal(array_out, array_in)

    # Check numpy outputs separately from other outputs
    expected_numpy_return_type = (
        has_numpy_dtype
        or return_type is np.ndarray
        or (isinstance(array_in, np.ndarray) and return_type is None)
    )
    expected_other_return_type = return_type in (tuple, list, None)

    if expected_numpy_return_type:
        # Test return type
        assert isinstance(array_out, np.ndarray)
        if as_any and input_type is pyvista_ndarray:
            assert type(array_out) is pyvista_ndarray
        else:
            assert type(array_out) is np.ndarray

        # Test dtype out
        assert array_out.dtype.type is np.dtype(dtype_out).type

        # Test copy
        is_same_type = isinstance(array_in, np.ndarray) and type(array_in) is type(array_out)
        is_same_dtype = isinstance(array_in, np.ndarray) and np.dtype(dtype_out) is array_in.dtype
        expect_copy = copy or not is_same_type or not is_same_dtype
        if expect_copy:
            assert array_out is not array_in
        else:
            assert array_out is array_in

    elif expected_other_return_type:
        assert isinstance(array_out, (int, float, tuple, list))

        ndim = np.array(array_in).ndim

        if ndim == 0:
            if dtype_out is not None:
                # Test scalars type/dtype
                assert isinstance(array_out, dtype_out)
            else:
                assert isinstance(array_out, dtype_in)

        elif return_type in (tuple, list):
            # Test sequence type
            assert isinstance(array_out, return_type)

            # Test sequence dtype
            if ndim == 1:
                assert isinstance(array_out[0], dtype_out)
            elif ndim == 2:
                assert isinstance(array_out[0][0], dtype_out)
            elif ndim == 3:
                assert isinstance(array_out[0][0][0], dtype_out)
            elif ndim == 4:
                assert isinstance(array_out[0][0][0][0], dtype_out)
            else:
                raise RuntimeError("Unexpected test case")

        # Test copy
        can_be_copied = deepcopy(array_in) is not array_in
        is_same_type = type(array_in) is type(array_out)
        change_dtype = dtype_out is not None and isinstance(array_in, (tuple, list))
        expect_copy = (copy and can_be_copied) or not is_same_type or change_dtype
        if expect_copy:
            assert array_in is not array_out
        else:
            assert array_in is array_out
    else:
        raise RuntimeError("Unexpected test case")

    # Test flags
    same_shape = np.shape(array_in) == np.shape(array_out)
    same_dtype = np.array(array_in).dtype == np.dtype(dtype_out)
    same_object = id(array_in) == id(array_out)
    same_type = type(array_in) is type(array_out)
    assert flags.same_shape == same_shape
    assert flags.same_dtype == same_dtype
    assert flags.same_object == same_object
    assert flags.same_type == same_type


def test_validate_array_return_type_raises():
    validate_array(0, return_type=None, dtype_out=np.float64)
    validate_array(0, return_type='numpy', dtype_out=np.float64)
    validate_array(0, return_type=np.ndarray, dtype_out=np.float64)
    validate_array(0, return_type=list, dtype_out=float)

    def msg(atype, dtype):
        return (
            f"Return type {atype} is not compatible with dtype_out={dtype}.\n"
            f"A list or tuple can only be returned if dtype_out is float, int, or bool."
        )

    atype, dtype = 'list', np.float64
    with pytest.raises(ValueError, match=msg(atype, dtype)):
        validate_array(0, return_type=atype, dtype_out=dtype)
    atype, dtype = list, np.float64
    with pytest.raises(ValueError, match=msg(atype, dtype)):
        validate_array(0, return_type=atype, dtype_out=dtype)
    atype, dtype = tuple, np.float64
    with pytest.raises(ValueError, match=msg(atype, dtype)):
        validate_array(0, return_type=atype, dtype_out=dtype)

    msg = (
        "Return type '0' is not valid. Return type must be one of: \n\t"
        "['numpy', 'list', 'tuple', <class 'numpy.ndarray'>, <class 'list'>, <class 'tuple'>]"
    )

    with pytest.raises(ValueError, match=escape(msg)):
        validate_array(0, return_type=0, dtype_out=float)


def test_validate_array_overflow_raises():
    msg = (
        "Cannot change dtype of Array from <class 'float'> to <class 'int'>.\n"
        "Float infinity cannot be converted to integer."
    )
    with pytest.raises(TypeError, match=msg):
        validate_array(float('inf'), must_be_finite=False, dtype_out=int)

    # no overflow raised
    overflowed = validate_array(np.array(np.inf), must_be_finite=False, dtype_out=np.int64)
    info = np.iinfo(overflowed.dtype)
    assert overflowed in (info.min, info.max)


@pytest.mark.parametrize('obj', [0, 0.0, "0"])
@pytest.mark.parametrize('classinfo', [int, (int, float), [int, float]])
@pytest.mark.parametrize('allow_subclass', [True, False])
@pytest.mark.parametrize('name', ["_input", "_object"])
def test_check_instance(obj, classinfo, allow_subclass, name):
    if isinstance(classinfo, list):
        with pytest.raises(TypeError):
            check_instance(obj, classinfo)
        return

    if allow_subclass:
        if isinstance(obj, classinfo):
            check_instance(obj, classinfo)
        else:
            with pytest.raises(TypeError, match='Object must be an instance of'):
                check_instance(obj, classinfo)
            with pytest.raises(TypeError, match=f'{name} must be an instance of'):
                check_instance(obj, classinfo, name=name)

    else:
        if type(classinfo) is tuple:
            if type(obj) in classinfo:
                check_type(obj, classinfo)
            else:
                with pytest.raises(TypeError, match=f'{name} must have one of the following types'):
                    check_type(obj, classinfo, name=name)
                with pytest.raises(TypeError, match='Object must have one of the following types'):
                    check_type(obj, classinfo)
        elif get_origin(classinfo) is Union:
            if type(obj) in get_args(classinfo):
                check_type(obj, classinfo)
            else:
                with pytest.raises(TypeError, match=f'{name} must have one of the following types'):
                    check_type(obj, classinfo, name=name)
                with pytest.raises(TypeError, match='Object must have one of the following types'):
                    check_type(obj, classinfo)
        else:
            if type(obj) is classinfo:
                check_type(obj, classinfo)
            else:
                with pytest.raises(TypeError, match=f'{name} must have type'):
                    check_type(obj, classinfo, name=name)
                with pytest.raises(TypeError, match='Object must have type'):
                    check_type(obj, classinfo)

    msg = "Name must be a string, got <class 'int'> instead."
    with pytest.raises(TypeError, match=msg):
        check_instance(0, int, name=0)


def test_check_type():
    check_type(0, int, name='abc')
    check_type(0, Union[int])
    with pytest.raises(TypeError):
        check_type("str", int)
    with pytest.raises(TypeError):
        check_type(0, int, name=1)
    check_type(0, Union[int, float])


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="Union type input requires python3.10 or higher",
)
def test_check_type_union():
    check_type(0, Union[int, float])


def test_check_string():
    check_string("abc")
    check_string("abc", name='123')
    msg = "Value must be an instance of <class 'str'>. Got <class 'int'> instead."
    with pytest.raises(TypeError, match=msg):
        check_string(0, name='Value')
    msg = "Object must be an instance of <class 'str'>. Got <class 'int'> instead."
    with pytest.raises(TypeError, match=msg):
        check_string(0)
    msg = "Name must be a string, got <class 'float'> instead."
    with pytest.raises(TypeError, match=msg):
        check_string("abc", name=0.0)

    class str_subclass(str):
        pass

    check_string(str_subclass(), allow_subclass=True)
    with pytest.raises(TypeError, match="Object must have type <class 'str'>."):
        check_string(str_subclass(), allow_subclass=False)


def test_check_less_than():
    check_less_than([0], 1)
    check_less_than(np.eye(3), 1, strict=False)
    msg = "Array values must all be less than 0."
    with pytest.raises(ValueError, match=msg):
        check_less_than(0, 0, strict=True)
    msg = "_input values must all be less than or equal to 0."
    with pytest.raises(ValueError, match=msg):
        check_less_than(1, 0, strict=False, name="_input")


def test_check_greater_than():
    check_greater_than([1], 0)
    check_greater_than(np.eye(3), 0, strict=False)
    msg = "Array values must all be greater than 0."
    with pytest.raises(ValueError, match=msg):
        check_greater_than(0, 0, strict=True)
    msg = "_input values must all be greater than or equal to 0."
    with pytest.raises(ValueError, match=msg):
        check_greater_than(-1, 0, strict=False, name="_input")


def test_check_real():
    check_real(1)
    check_real(-2.0)
    check_real(np.array(2.0, dtype="uint8"))
    check_real(np.array(True, dtype=bool))
    msg = 'Array must have real numbers.'
    with pytest.raises(TypeError, match=msg):
        check_real(1 + 1j)
    msg = '_input must have real numbers.'
    with pytest.raises(TypeError, match=msg):
        check_real(1 + 1j, name="_input")


def test_check_finite():
    check_finite(0)
    msg = '_input must have finite values.'
    with pytest.raises(ValueError, match=msg):
        check_finite(np.nan, name="_input")


def test_check_integerlike():
    check_integer(1)
    check_integer([2, 3.0])
    msg = "Input has incorrect dtype of <class 'float'>. The dtype must be a subtype of <class 'numpy.integer'>."
    with pytest.raises(TypeError, match=msg):
        check_integer([2, 3.0], strict=True, name="_input")
    msg = "_input must have integer-like values."
    with pytest.raises(ValueError, match=msg):
        check_integer([2, 3.4], strict=False, name="_input")


def test_check_sequence():
    check_sequence((1,), name='abc')
    check_sequence(range(3))
    check_sequence("abc")
    with pytest.raises(TypeError, match="_input"):
        check_sequence(np.array(1), name="_input")


def test_check_iterable():
    check_iterable((1,), name='abc')
    check_iterable(range(3))
    check_iterable("abc")
    check_iterable(np.array(1))
    with pytest.raises(TypeError, match="_input"):
        check_iterable(1, name="_input")


def test_check_length():
    check_length((1,))
    check_length(
        [
            1,
        ],
    )
    check_length(np.ndarray((1,)))
    check_length((1,), exact_length=1, min_length=1, max_length=1, must_be_1d=True)
    check_length((1,), exact_length=[1, 2.0])

    with pytest.raises(ValueError, match="'exact_length' must have integer-like values."):
        check_length((1,), exact_length=(1, 2.4), name="_input")

    msg = '_input must have a length equal to any of: 1. Got length 2 instead.'
    with pytest.raises(ValueError, match=msg):
        check_length((1, 2), exact_length=1, name="_input")
    msg = '_input must have a length equal to any of: [3, 4]. Got length 2 instead.'
    with pytest.raises(ValueError, match=escape(msg)):
        check_length((1, 2), exact_length=[3, 4], name="_input")

    msg = "_input must have a maximum length of 1. Got length 2 instead."
    with pytest.raises(ValueError, match=msg):
        check_length((1, 2), max_length=1, name="_input")

    msg = "_input must have a minimum length of 2. Got length 1 instead."
    with pytest.raises(ValueError, match=msg):
        check_length((1,), min_length=2, name="_input")

    msg = 'Range with 2 elements must be sorted in ascending order. Got:\n    (4, 2)'
    with pytest.raises(ValueError, match=escape(msg)):
        check_length(
            (
                1,
                2,
                3,
            ),
            min_length=4,
            max_length=2,
        )

    msg = 'Array has the incorrect number of dimensions. Got 2, expected 1.'
    with pytest.raises(ValueError, match=escape(msg)):
        check_length(((1, 2), (3, 4)), must_be_1d=True)

    with pytest.raises(TypeError, match="object of type 'int' has no len()"):
        check_length(0, allow_scalar=False)


def test_check_nonnegative():
    check_nonnegative(0)
    check_nonnegative(np.eye(3))
    msg = "Array values must all be greater than or equal to 0."
    with pytest.raises(ValueError, match=msg):
        check_nonnegative(-1)


@pytest.mark.parametrize('shape', [(), (8,), (4, 6), (2, 3, 4)])
@pytest.mark.parametrize('axis', [None, -1, -2, -3, 0, 1, 2, 3])
@pytest.mark.parametrize('ascending', [True, False])
@pytest.mark.parametrize('strict', [True, False])
@pytest.mark.parametrize('as_list', [True, False])
def test_check_sorted(shape, axis, ascending, strict, as_list):
    def _check_sorted_params(arr):
        check_sorted(arr, axis=axis, strict=strict, ascending=ascending)

    if shape == ():
        # test always succeeds with scalar
        _check_sorted_params(0)
        return

    # Create ascending array with unique values
    num_elements = np.prod(shape)
    arr_strict_ascending = np.arange(num_elements).reshape(shape)

    # needed to support numpy <1.25
    # needed to support vtk 9.0.3
    # check for removal when support for vtk 9.0.3 is removed
    try:
        AxisError = np.exceptions.AxisError
    except AttributeError:
        AxisError = np.AxisError

    try:
        # Create ascending array with duplicate values
        arr_ascending = np.repeat(arr_strict_ascending, 2, axis=axis)
        # Create descending arrays
        arr_descending = np.flip(arr_ascending, axis=axis)
        arr_strict_descending = np.flip(arr_strict_ascending, axis=axis)
    except AxisError:
        # test ValueError is raised whenever an AxisError would otherwise be raised
        with pytest.raises(
            ValueError,
            match=f'Axis {axis} is out of bounds for ndim {arr_strict_ascending.ndim}',
        ):
            _check_sorted_params(arr_strict_ascending)
        return

    arr_ascending = arr_ascending.tolist() if as_list else arr_ascending
    arr_strict_ascending = arr_strict_ascending.tolist() if as_list else arr_strict_ascending
    arr_descending = arr_descending.tolist() if as_list else arr_descending
    arr_strict_descending = arr_strict_descending.tolist() if as_list else arr_strict_descending

    if axis is None and not as_list and arr_ascending.ndim > 1:
        # test that axis=None will flatten array and cause it not to be sorted for higher dimension arrays
        with pytest.raises(ValueError):  # noqa: PT011
            _check_sorted_params(arr_ascending)
        return

    if strict and ascending:
        _check_sorted_params(arr_strict_ascending)
        for a in [arr_ascending, arr_descending, arr_strict_descending]:
            with pytest.raises(ValueError, match="must be sorted in strict ascending order"):
                _check_sorted_params(a)

    elif not strict and ascending:
        _check_sorted_params(arr_ascending)
        _check_sorted_params(arr_strict_ascending)
        for a in [arr_descending, arr_strict_descending]:
            with pytest.raises(ValueError, match="must be sorted in ascending order"):
                _check_sorted_params(a)

    elif strict and not ascending:
        _check_sorted_params(arr_strict_descending)
        for a in [arr_ascending, arr_strict_ascending, arr_descending]:
            with pytest.raises(ValueError, match="must be sorted in strict descending order"):
                _check_sorted_params(a)

    elif not strict and not ascending:
        _check_sorted_params(arr_descending)
        _check_sorted_params(arr_strict_descending)
        for a in [arr_ascending, arr_strict_ascending]:
            with pytest.raises(ValueError, match="must be sorted in descending order"):
                _check_sorted_params(a)


def test_check_sorted_error_repr():
    array = np.zeros(shape=(10, 10))
    msg = "Array with 100 elements must be sorted in strict ascending order. Got:\n    array([[0., 0... 0., 0., 0.]])"
    with pytest.raises(ValueError, match=escape(msg)):
        check_sorted(array, ascending=True, strict=True)


def test_check_iterable_items():
    check_iterable_items([1, 2, 3], int)
    check_iterable_items(("a", "b", "c"), str)
    check_iterable_items("abc", str)
    check_iterable_items(range(10), int)
    msg = "All items of Iterable must be an instance of <class 'str'>. Got <class 'int'> instead."
    with pytest.raises(TypeError, match=escape(msg)):
        check_iterable_items(["abc", 1], str)
    with pytest.raises(TypeError, match="All items of _input"):
        check_iterable_items(["abc", 1], str, name="_input")


@pytest.mark.parametrize('number', [1, 1.0, True, 1 + 1j])
@pytest.mark.parametrize('definition', ['builtin', 'numpy'])
@pytest.mark.parametrize('must_be_real', [True, False])
def test_check_number(number, definition, must_be_real):
    if definition == 'numpy':
        number = np.array([number])[0]

    if isinstance(number, np.bool_) or (not isinstance(number, Real) and must_be_real):
        # Test bool types always raise an error
        # Test complex types raise an error when `must_be_real` is True
        with pytest.raises(TypeError):
            check_number(number, must_be_real=must_be_real)
    else:
        # All other cases should succeed
        check_number(number, must_be_real=must_be_real)

    if definition == 'numpy':
        # Test numpy types raise an error when definition is 'builtin'
        if isinstance(number, float) or (isinstance(number, complex) and not must_be_real):
            # np.float_ and np.complex_ subclass float and complex, respectively,
            # so no error is raised
            check_number(number, must_be_real=must_be_real, definition='builtin')
        else:
            with pytest.raises(TypeError):
                check_number(number, must_be_real=must_be_real, definition='builtin')
    elif definition == 'builtin':
        # Test builtin types raise an error when definition is 'numpy'
        with pytest.raises(TypeError):
            check_number(number, must_be_real=must_be_real, definition='numpy')


def test_check_number_raises():
    msg = (
        "_input must be an instance of <class 'numbers.Real'>. Got <class 'numpy.ndarray'> instead."
    )
    with pytest.raises(TypeError, match=msg):
        check_number(np.array(0), name='_input')
    msg = "Object must be"
    with pytest.raises(TypeError, match=msg):
        check_number(np.array(0))
    msg = "Object must be"
    with pytest.raises(TypeError, match=msg):
        check_number(1 + 1j, must_be_real=True)


def test_check_contains():
    check_contains(["foo", "bar"], must_contain="foo")
    msg = "Input 'foo' is not valid. Input must be one of: \n\t['cat', 'bar']"
    with pytest.raises(ValueError, match=escape(msg)):
        check_contains(["cat", "bar"], must_contain="foo")
    msg = "_input '5' is not valid. _input must be in: \n\trange(0, 4)"
    with pytest.raises(ValueError, match=escape(msg)):
        check_contains(range(4), must_contain=5, name="_input")


@pytest.mark.parametrize('name', ['_input', 'Axes'])
def test_validate_axes(name):
    axes_right = np.eye(3)
    axes_left = np.array([[1, 0.0, 0], [0, 1, 0], [0, 0, -1]])

    # test different input args
    axes = validate_axes(axes_right)
    assert np.array_equal(axes, axes_right)
    axes = validate_axes(
        [[1], [0], [0]],
        [[0, 1, 0]],
        must_have_orientation='right',
        must_be_orthogonal=True,
    )
    assert np.array_equal(axes, axes_right)
    axes = validate_axes([1, 0, 0], [[0, 1, 0]], (0, 0, 1))
    assert np.array_equal(axes, axes_right)
    assert np.issubdtype(axes.dtype, np.floating)

    axes = validate_axes(np.eye(3).astype(int))
    assert np.array_equal(axes, axes_right)
    assert np.issubdtype(axes.dtype, np.floating)

    # test with non-identity orthogonal axes
    from pyvista.core.utilities.transformations import axis_angle_rotation

    axes = axis_angle_rotation(axis=(1, 2, 3), angle=(30))[:3, :3]
    _ = validate_axes(axes)

    # test bad input
    with pytest.raises(ValueError, match=f"{name} cannot be parallel."):
        validate_axes([[1, 0, 0], [1, 0, 0], [0, 1, 0]], name=name)
    with pytest.raises(ValueError, match=f"{name} cannot be parallel."):
        validate_axes([[1, 2, 3], [2, 4, 6], [0, 1, 0]], name=name)
    with pytest.raises(ValueError, match="Axes cannot be parallel."):
        validate_axes([[0, 1, 0], [1, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError, match=f"{name} cannot be zeros."):
        validate_axes([[1, 0, 0], [0, 1, 0], [0, 0, 0]], name=name)
    with pytest.raises(ValueError, match="Axes cannot be zeros."):
        validate_axes([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match="Axes cannot be zeros."):
        validate_axes([[0, 0, 0], [0, 1, 0], [0, 0, 1]])

    # test normalize
    axes_scaled = axes_right * 2
    axes = validate_axes(axes_scaled, normalize=False)
    assert np.array_equal(axes, axes_scaled)
    axes = validate_axes(axes_scaled, normalize=True)
    assert np.array_equal(axes, axes_right)

    # test orientation
    validate_axes([1, 0, 0], [0, 1, 0], must_have_orientation='left')
    validate_axes(axes_left, must_have_orientation=None)
    validate_axes(axes_left, must_have_orientation='left')
    with pytest.raises(ValueError, match=f"{name} do not have a right-handed orientation."):
        validate_axes(axes_left, must_have_orientation='right', name=name)

    validate_axes(axes_right, must_have_orientation=None)
    validate_axes(axes_right, must_have_orientation='right')
    with pytest.raises(ValueError, match=f"{name} do not have a left-handed orientation."):
        validate_axes(axes_right, must_have_orientation='left', name=name)

    # test specifying two vectors without orientation raises error (3rd cannot be computed)
    with pytest.raises(
        ValueError,
        match=f"{name} orientation must be specified when only two vectors are given.",
    ):
        validate_axes([1, 0, 0], [0, 1, 0], must_have_orientation=None, name=name)

    msg = "Axes has shape (3,) which is not allowed. Shape must be one of [(2, 3), (3, 3)]."
    with pytest.raises(ValueError, match=escape(msg)):
        validate_axes([1, 0, 0])

    msg = (
        'Incorrect number of axes arguments. Number of arguments must be either:\n'
        '\tOne arg (a single array with two or three vectors),'
        '\tTwo args (two vectors), or'
        '\tThree args (three vectors).'
    )
    with pytest.raises(ValueError, match=escape(msg)):
        validate_axes()
    with pytest.raises(ValueError, match=escape(msg)):
        validate_axes(0, 0, 0, 0)


@pytest.mark.parametrize('bias_index', [(0, 1), (1, 0), (2, 0)])
def test_validate_axes_orthogonal(bias_index):
    axes_right = np.eye(3)
    axes_right[bias_index[0], bias_index[1]] = 0.1
    axes_left = np.array([[1, 0.0, 0], [0, 1, 0], [0, 0, -1]])
    axes_left[bias_index[0], bias_index[1]] = 0.1

    msg = "Axes are not orthogonal."
    axes = validate_axes(
        axes_right,
        must_be_orthogonal=False,
        normalize=False,
        must_have_orientation='right',
    )
    assert np.array_equal(axes, axes_right)
    with pytest.raises(ValueError, match=msg):
        validate_axes(axes_right, must_be_orthogonal=True)

    axes = validate_axes(
        axes_left,
        must_be_orthogonal=False,
        normalize=False,
        must_have_orientation='left',
    )
    assert np.array_equal(axes, axes_left)
    with pytest.raises(ValueError, match=msg):
        validate_axes(axes_left, must_be_orthogonal=True)


@pytest.mark.parametrize('as_any', [True, False])
@pytest.mark.parametrize('copy', [True, False])
@pytest.mark.parametrize('dtype', [None, float])
def test_cast_to_numpy(as_any, copy, dtype):
    array_in = pyvista_ndarray([1, 2])
    array_out = _cast_to_numpy(array_in, copy=copy, as_any=as_any, dtype=dtype)
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


def test_cast_to_numpy_raises():
    if sys.version_info < (3, 9) and sys.platform == 'linux':
        err = TypeError
        msg = "Object arrays are not supported."
    else:
        err = ValueError
        msg = "Array cannot be cast as <class 'numpy.ndarray'>."
    with pytest.raises(err, match=msg):
        _cast_to_numpy([[1], [2, 3]])

    msg = "Object arrays are not supported."
    with pytest.raises(TypeError, match=msg):
        _cast_to_numpy(list)


def test_cast_to_numpy_must_be_real():
    _ = _cast_to_numpy([0, 1], must_be_real=True)
    _ = _cast_to_numpy("abc", must_be_real=False)

    msg = "Array must have real numbers. Got dtype <class 'numpy.complex128'>"
    with pytest.raises(TypeError, match=msg):
        _ = _cast_to_numpy([0, 1 + 1j], must_be_real=True)
    msg = "Array must have real numbers. Got dtype <class 'numpy.str_'>"
    with pytest.raises(TypeError, match=msg):
        _ = _cast_to_numpy("abc", must_be_real=True)


def test_cast_to_tuple():
    array_in = np.zeros(shape=(2, 2, 3))
    array_tuple = _cast_to_tuple(array_in)
    assert array_tuple == (((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    array_list = array_in.tolist()
    assert np.array_equal(array_tuple, array_list)


def test_cast_to_list():
    array_in = np.zeros(shape=(3, 4, 5))
    array_list = _cast_to_list(array_in)
    assert np.array_equal(array_in, array_list)


@pytest.mark.parametrize(
    ('cls', 'shape'),
    [
        (vtkMatrix3x3, (3, 3)),
        (vtkMatrix4x4, (4, 4)),
    ],
)
def test_array_from_vtkmatrix(cls, shape):
    expected = np.random.default_rng().random(shape)
    mat = cls()
    for i, j in itertools.product(range(shape[0]), range(shape[1])):
        mat.SetElement(i, j, expected[i, j])
    actual = _array_from_vtkmatrix(mat, shape=shape)
    assert np.array_equal(actual, expected)

    # Test this matches public function
    expected = array_from_vtkmatrix(mat)
    assert np.array_equal(actual, expected)


arraylike_shapes = [
    (),
    (0,),
    (1,),
    (
        1,
        0,
    ),
    (1, 1, 0),
    (1, 1, 1, 0),
    (
        1,
        2,
    ),
    (1, 2, 3),
    (
        1,
        2,
        3,
        4,
    ),
]

ArrayLikePropsTuple = namedtuple(  # noqa: PYI024
    'ArrayLikePropsTuple',
    ['array', 'shape', 'dtype', 'ndim', 'size', 'wrapper', 'return_original'],
)


class arraylike_types(Enum):
    Number = auto()
    NumpyArraySequence = auto()
    NumberSequence1D = auto()
    NumberSequence2D = auto()
    NumpyArray = auto()


@pytest.mark.parametrize('arraylike_type', arraylike_types)
@pytest.mark.parametrize('shape_in', arraylike_shapes)
@pytest.mark.parametrize('dtype_in', [float, int, bool, np.float64, np.int_, np.bool_, np.uint8])
def test_array_wrappers(arraylike_type, shape_in, dtype_in):
    # Skip tests for impossible scalar cases
    is_scalar = shape_in == ()
    is_sequence_type = arraylike_type in [
        arraylike_types.NumpyArraySequence,
        arraylike_types.NumberSequence1D,
        arraylike_types.NumberSequence2D,
    ]
    is_scalar_type = arraylike_type is arraylike_types.Number
    if (is_scalar and is_sequence_type) or (not is_scalar and is_scalar_type):
        pytest.skip("Scalar cannot be a sequence")

    # Skip tests for sequences of numpy dtypes
    # This is done since sequences are generated using `array.tolist()`
    # which will cast numpy dtypes to builtin types (which are tested separately)
    if arraylike_type in [
        arraylike_types.NumberSequence1D,
        arraylike_types.NumberSequence2D,
    ] and issubclass(dtype_in, np.generic):
        pytest.skip("No tests for sequences of numpy dtypes.")

    # Set up test array and keep track of special empty sequence cases
    if is_scalar:
        initial_array = dtype_in(0)
        is_empty = False
    else:
        initial_array = np.zeros(shape=shape_in, dtype=dtype_in)
        is_empty = initial_array.shape[-1] == 0

    if arraylike_type is arraylike_types.Number:
        array_before_wrap = initial_array
        expected = ArrayLikePropsTuple(
            array=array_before_wrap,
            shape=shape_in,
            dtype=dtype_in,
            ndim=0,
            size=1,
            wrapper=_NumberWrapper,
            return_original=True,
        )
    elif arraylike_type is arraylike_types.NumpyArray:
        array_before_wrap = np.array(initial_array)
        expected = ArrayLikePropsTuple(
            array=array_before_wrap,
            shape=shape_in,
            dtype=np.dtype(dtype_in),
            ndim=array_before_wrap.ndim,
            size=array_before_wrap.size,
            wrapper=_NumpyArrayWrapper,
            return_original=True,
        )
    elif arraylike_type is arraylike_types.NumpyArraySequence:
        # convert to list array and replace items with numpy arrays
        depth = initial_array.ndim
        if depth == 4:
            array_before_wrap = [[[[initial_array]]]]
            shape_out = (1, 1, 1, 1, *shape_in)
        elif depth == 3:
            array_before_wrap = [[[initial_array]]]
            shape_out = (1, 1, 1, *shape_in)
        elif depth == 2:
            array_before_wrap = [[initial_array]]
            shape_out = (1, 1, *shape_in)
        elif depth == 1:
            array_before_wrap = [initial_array]
            shape_out = (1, *shape_in)
        else:
            raise RuntimeError('Unexpected test case.')

        expected = ArrayLikePropsTuple(
            array=np.array(array_before_wrap),
            shape=shape_out,
            dtype=np.array(array_before_wrap).dtype,
            ndim=np.array(array_before_wrap).ndim,
            size=np.array(array_before_wrap).size,
            wrapper=_NumpyArrayWrapper,
            return_original=False,
        )

    elif arraylike_type in [arraylike_types.NumberSequence1D, arraylike_types.NumberSequence2D]:
        if is_empty:
            # Cannot infer dtype from an empty sequence at runtime,
            # so we assume the dtype is float by default
            dtype_out = float
            # Check this matches default numpy behavior
            assert np.array([()]).dtype.type is np.float64
        else:
            dtype_out = dtype_in
        depth = initial_array.ndim
        array_before_wrap = initial_array.tolist()
        if depth in (1, 2):
            wrapper = _SequenceWrapper if depth == 1 else _NestedSequenceWrapper

            # sequence is expected as-is
            expected = ArrayLikePropsTuple(
                array=array_before_wrap,
                shape=shape_in,
                dtype=dtype_out,
                ndim=depth,
                size=initial_array.size,
                wrapper=wrapper,
                return_original=True,
            )
        else:
            # cast to a numpy array
            expected = ArrayLikePropsTuple(
                array=np.asarray(array_before_wrap),
                shape=shape_in,
                dtype=dtype_out,
                ndim=depth,
                size=initial_array.size,
                wrapper=_NumpyArrayWrapper,
                return_original=False,
            )
    else:
        raise RuntimeError("Unexpected test case.")

    # Test abstract wrapper
    wrapped_abstract = _ArrayLikeWrapper(array_before_wrap)
    assert np.array_equal(wrapped_abstract._array, expected.array)
    assert wrapped_abstract.shape == expected.shape
    assert wrapped_abstract.dtype == expected.dtype
    assert wrapped_abstract.ndim == expected.ndim
    assert wrapped_abstract.size == expected.size
    assert type(wrapped_abstract) is expected.wrapper

    # Test child wrapper
    wrapped_child = expected.wrapper(array_before_wrap)
    assert np.array_equal(wrapped_child._array, expected.array)
    assert wrapped_child.shape == expected.shape
    assert wrapped_child.dtype == expected.dtype
    assert wrapped_child.ndim == expected.ndim
    assert wrapped_abstract.size == expected.size
    assert type(wrapped_child) is expected.wrapper

    # Test wrapping self returns self
    wrapped_wrapped = expected.wrapper(wrapped_child)
    assert wrapped_wrapped is wrapped_child

    assert np.array_equal(wrapped_wrapped._array, expected.array)
    assert wrapped_wrapped.shape == expected.shape
    assert wrapped_wrapped.dtype == expected.dtype
    assert wrapped_wrapped.ndim == expected.ndim
    assert wrapped_abstract.size == expected.size
    assert type(wrapped_wrapped) is expected.wrapper

    assert repr(wrapped_wrapped) == f"{expected.wrapper.__name__}({wrapped_wrapped._array!r})"


ragged_arrays = (
    [[1, 2, 3], [4, 5], [6, 7, 8, 9]],
    [np.array([1, 2, 3]), np.array([4, 5]), np.array([6, 7, 8, 9])],
)


@pytest.mark.skipif(
    sys.platform == 'linux' and sys.version_info < (3, 9, 0),
    reason="Numpy raise a warning, not an error.",
)
@pytest.mark.parametrize('ragged_array', ragged_arrays)
def test_array_wrapper_ragged_array(ragged_array):
    # assert casting directly to numpy array raises error
    with pytest.raises(ValueError, match='inhomogeneous shape'):
        np.array(ragged_array)

    msg = 'The following array is not valid:\n\t['
    with pytest.raises(ValueError, match=escape(msg)):
        _ArrayLikeWrapper(ragged_array)


def _get_default_kwargs(call: Callable) -> Dict[str, Any]:
    """Get all args/kwargs and their default value"""
    params = dict(inspect.signature(call).parameters)
    # Get default value for positional or keyword args
    return {
        key: val.default
        for key, val in params.items()
        if val.kind is inspect.Parameter.KEYWORD_ONLY
    }


@pytest.fixture()
def validate_array_kwargs() -> Dict[str, Any]:
    return _get_default_kwargs(validate_array)


@pytest.mark.parametrize(
    'func',
    [validate_number, validate_array3, validate_arrayNx3, validate_arrayN, validate_data_range],
)
def test_validate_array_specialized_kwargs(func, validate_array_kwargs):
    """Test kwargs for functions which call validate_array

    This test is used to ensure specialized validate_array functions do
    not modify the values of kwargs which are intended to be passed-through
    """

    actual_kwargs = _get_default_kwargs(func)
    expected_kwargs = validate_array_kwargs

    if func is validate_number:
        # Remove unused kwargs
        expected_kwargs.pop('must_have_shape')
        expected_kwargs.pop('reshape_to')
        expected_kwargs.pop('broadcast_to')
        expected_kwargs.pop('return_type')
        expected_kwargs.pop('must_be_sorted')
        expected_kwargs.pop('must_have_length')
        expected_kwargs.pop('must_have_max_length')
        expected_kwargs.pop('must_have_min_length')
        expected_kwargs.pop('as_any')
        expected_kwargs.pop('copy')

        # Change default values
        assert expected_kwargs['must_be_finite'] is not True
        expected_kwargs['must_be_finite'] = True
        assert expected_kwargs['name'] != 'Number'
        expected_kwargs['name'] = 'Number'

        # Remove wrapper-specific kwargs
        actual_kwargs.pop('reshape')

    elif func is validate_array3:
        # Remove unused kwargs
        expected_kwargs.pop('must_have_shape')
        expected_kwargs.pop('reshape_to')
        expected_kwargs.pop('broadcast_to')
        expected_kwargs.pop('return_type')
        expected_kwargs.pop('must_have_length')
        expected_kwargs.pop('must_have_max_length')
        expected_kwargs.pop('must_have_min_length')

        # Remove wrapper-specific kwargs
        actual_kwargs.pop('reshape')
        actual_kwargs.pop('broadcast')

    elif func is validate_data_range:
        # Remove unused kwargs
        expected_kwargs.pop('must_have_shape')
        expected_kwargs.pop('reshape_to')
        expected_kwargs.pop('broadcast_to')
        expected_kwargs.pop('return_type')
        expected_kwargs.pop('must_have_length')
        expected_kwargs.pop('must_have_max_length')
        expected_kwargs.pop('must_have_min_length')
        expected_kwargs.pop('must_be_sorted')

        assert expected_kwargs['name'] != 'Data Range'
        expected_kwargs['name'] = 'Data Range'

    elif func is validate_arrayNx3 or validate_arrayN or validate_arrayN_unsigned:
        # Remove unused kwargs
        expected_kwargs.pop('must_have_shape')
        expected_kwargs.pop('reshape_to')
        expected_kwargs.pop('broadcast_to')
        expected_kwargs.pop('return_type')

        # Remove wrapper-specific kwargs
        actual_kwargs.pop('reshape')

        if func is validate_arrayN_unsigned:
            expected_kwargs.pop('must_be_finite')
            expected_kwargs.pop('must_be_integer')
            expected_kwargs.pop('must_be_nonnegative')

    assert actual_kwargs == expected_kwargs
