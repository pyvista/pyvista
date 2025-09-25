from __future__ import annotations

import itertools
from re import escape
import sys
from typing import NamedTuple
from typing import Union
from typing import get_args
from typing import get_origin

import numpy as np
import pytest
import scipy
from vtk import vtkTransform

from pyvista.core import pyvista_ndarray
from pyvista.core._validation import check_contains
from pyvista.core._validation import check_finite
from pyvista.core._validation import check_greater_than
from pyvista.core._validation import check_instance
from pyvista.core._validation import check_integer
from pyvista.core._validation import check_iterable
from pyvista.core._validation import check_iterable_items
from pyvista.core._validation import check_length
from pyvista.core._validation import check_less_than
from pyvista.core._validation import check_ndim
from pyvista.core._validation import check_nonnegative
from pyvista.core._validation import check_number
from pyvista.core._validation import check_range
from pyvista.core._validation import check_real
from pyvista.core._validation import check_sequence
from pyvista.core._validation import check_shape
from pyvista.core._validation import check_sorted
from pyvista.core._validation import check_string
from pyvista.core._validation import check_subdtype
from pyvista.core._validation import check_type
from pyvista.core._validation import validate_array
from pyvista.core._validation import validate_array3
from pyvista.core._validation import validate_arrayN
from pyvista.core._validation import validate_arrayN_unsigned
from pyvista.core._validation import validate_arrayNx3
from pyvista.core._validation import validate_axes
from pyvista.core._validation import validate_data_range
from pyvista.core._validation import validate_dimensionality
from pyvista.core._validation import validate_number
from pyvista.core._validation import validate_rotation
from pyvista.core._validation import validate_transform3x3
from pyvista.core._validation import validate_transform4x4
from pyvista.core._validation._cast_array import _cast_to_list
from pyvista.core._validation._cast_array import _cast_to_numpy
from pyvista.core._validation._cast_array import _cast_to_tuple
from pyvista.core._validation.check import _validate_shape_value
from pyvista.core._validation.validate import _array_from_vtkmatrix
from pyvista.core._validation.validate import _set_default_kwarg_mandatory
from pyvista.core._validation.validate import _validate_color_sequence
from pyvista.core._vtk_core import vtkMatrix3x3
from pyvista.core._vtk_core import vtkMatrix4x4
from pyvista.core.utilities.arrays import array_from_vtkmatrix
from pyvista.core.utilities.arrays import vtkmatrix_from_array
from tests.conftest import NUMPY_VERSION_INFO


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
    with pytest.raises(ValueError, match=escape('Shape must be one of [(3, 3), (4, 4)].')):
        validate_transform4x4(np.array([1, 2, 3]))
    with pytest.raises(TypeError, match='Input transform must be one of'):
        validate_transform4x4('abc')


@pytest.mark.parametrize(
    'transform_like',
    [
        np.eye(3),
        np.eye(3).tolist(),
        vtkmatrix_from_array(np.eye(3)),
        scipy.spatial.transform.Rotation.from_matrix(np.eye(3)),
    ],
    ids=['numpy', 'list', 'vtk', 'scipy'],
)
def test_validate_transform3x3(transform_like):
    result = validate_transform3x3(transform_like)
    assert type(result) is np.ndarray
    assert np.array_equal(result, np.eye(3))


def test_validate_transform3x3_raises():
    match = (
        'Input transform must be one of:'
        '\n\tvtkMatrix3x3'
        '\n\t3x3 np.ndarray'
        '\n\tscipy.spatial.transform.Rotation'
        "\nGot array([1, 2, 3]) with type <class 'numpy.ndarray'> instead."
    )
    with pytest.raises(TypeError, match=escape(match)):
        validate_transform3x3(np.array([1, 2, 3]))
    match = (
        'Input transform must be one of:'
        '\n\tvtkMatrix3x3'
        '\n\t3x3 np.ndarray'
        '\n\tscipy.spatial.transform.Rotation'
        "\nGot 'abc' with type <class 'str'> instead."
    )
    with pytest.raises(TypeError, match=match):
        validate_transform3x3('abc')


def test_check_subdtype():
    check_subdtype(int, np.integer)
    check_subdtype(np.dtype(int), np.integer)
    check_subdtype(np.array([1, 2, 3]), np.integer)
    check_subdtype(np.array([1.0, 2, 3]), float)
    check_subdtype(np.array([1.0, 2, 3], dtype='uint8'), 'uint8')
    check_subdtype(np.array([1.0, 2, 3]), ('uint8', float))
    match = "Input has incorrect dtype of 'int32'. The dtype must be a subtype of <class 'float'>."
    with pytest.raises(TypeError, match=match):
        check_subdtype(np.array([1, 2, 3]).astype('int32'), float)
    match = (
        "Input has incorrect dtype of 'complex128'. The dtype must be a subtype of at least "
        "one of \n(<class 'numpy.integer'>, <class 'numpy.floating'>)."
    )
    with pytest.raises(TypeError, match=escape(match)):
        check_subdtype(np.array([1 + 1j, 2, 3]), (np.integer, np.floating))


def test_validate_number():
    validate_number([2.0])
    num = validate_number(1)
    assert num == 1
    assert isinstance(num, int)

    num = validate_number(2.0, to_list=False, must_have_shape=(), reshape=False)
    assert num == 2.0
    assert type(num) is np.ndarray
    assert num.dtype.type is np.float64

    match = (
        "Parameter 'must_have_shape' cannot be set for function `validate_number`.\n"
        'Its value is automatically set to `()`.'
    )
    with pytest.raises(ValueError, match=escape(match)):
        validate_number(1, must_have_shape=2, reshape=False)


def test_validate_data_range():
    rng = validate_data_range([0, 1])
    assert rng == (0, 1)

    rng = validate_data_range((0, 2.5), to_list=True)
    assert rng == [0.0, 2.5]

    rng = validate_data_range((-10, -10), to_tuple=False, must_have_shape=2)
    assert type(rng) is np.ndarray

    match = 'Data Range with 2 elements must be sorted in ascending order. Got:\n    array([1, 0])'
    with pytest.raises(ValueError, match=escape(match)):
        validate_data_range((1, 0))

    match = (
        "Parameter 'must_have_shape' cannot be set for function `validate_data_range`.\n"
        'Its value is automatically set to `2`.'
    )
    with pytest.raises(ValueError, match=match):
        validate_data_range((0, 1), must_have_shape=3)


def test_set_default_kwarg_mandatory():
    default_value = 1
    default_key = 'k'

    # Test parameter unset
    kwargs = {}
    _set_default_kwarg_mandatory(kwargs, default_key, default_value)
    assert kwargs[default_key] == default_value

    # Test parameter already set to default
    kwargs = {}
    kwargs[default_key] = default_value
    _set_default_kwarg_mandatory(kwargs, default_key, default_value)
    assert kwargs[default_key] == default_value

    # Test parameter set to non-default
    kwargs = {}
    kwargs[default_key] = default_value * 2
    match = (
        "Parameter 'k' cannot be set for function `test_set_default_kwarg_mandatory`.\n"
        'Its value is automatically set to `1`.'
    )
    with pytest.raises(ValueError, match=match):
        _set_default_kwarg_mandatory(kwargs, default_key, default_value)


def test_check_shape():
    check_shape(0, ())
    check_shape(0, [(), 2])
    check_shape((1, 2, 3), [(), 3])
    check_shape((1, 2, 3), [-1])
    check_shape((1, 2, 3), -1)

    match = 'Input has shape (3,) which is not allowed. Shape must be 0.'
    with pytest.raises(ValueError, match=escape(match)):
        check_shape((1, 2, 3), 0, name='Input')

    match = 'Array has shape (3,) which is not allowed. Shape must be one of [(), (4, 5)].'
    with pytest.raises(ValueError, match=escape(match)):
        check_shape((1, 2, 3), [(), (4, 5)])


def test_check_ndim():
    check_ndim(0, 0)
    check_ndim(np.array(0), 0)
    check_ndim((1, 2, 3), range(2))
    check_ndim([[1, 2, 3]], (0, 2))

    match = 'Input has the incorrect number of dimensions. Got 1, expected 0.'
    with pytest.raises(ValueError, match=escape(match)):
        check_ndim((1, 2, 3), 0, name='Input')

    match = 'Array has the incorrect number of dimensions. Got 1, expected one of [4, 5].'
    with pytest.raises(ValueError, match=escape(match)):
        check_ndim((1, 2, 3), [4, 5])


def test_validate_shape_value():
    match = '`None` is not a valid shape. Use `()` instead.'
    with pytest.raises(TypeError, match=escape(match)):
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

    match = (
        "Shape must be an instance of any type (<class 'int'>, <class 'tuple'>). "
        "Got <class 'float'> instead."
    )
    with pytest.raises(TypeError, match=escape(match)):
        _validate_shape_value(1.0)

    match = 'Shape values must all be greater than or equal to -1.'
    with pytest.raises(ValueError, match=match):
        _validate_shape_value(-2)

    match = "All items of Shape must be an instance of <class 'int'>. Got <class 'tuple'> instead."
    with pytest.raises(TypeError, match=match):
        _validate_shape_value(((1, 2), (3, 4)))


@pytest.mark.parametrize('reshape', [True, False])
def test_validate_arrayNx3(reshape):  # noqa: N802
    arr = validate_arrayNx3((1, 2, 3))
    assert arr.shape == (1, 3)
    assert np.array_equal(arr, [[1, 2, 3]])

    if not reshape:
        match = 'Array has shape (3,) which is not allowed. Shape must be (-1, 3).'
        with pytest.raises(ValueError, match=escape(match)):
            validate_arrayNx3((1, 2, 3), reshape=False)

    arr = validate_arrayNx3([(1, 2, 3), (4, 5, 6)], reshape=reshape)
    assert arr.shape == (2, 3)

    match = (
        "Parameter 'must_have_shape' cannot be set for function `validate_arrayNx3`.\n"
        'Its value is automatically set to `[3, (-1, 3)]`.'
    )
    with pytest.raises(ValueError, match=escape(match)):
        validate_arrayNx3((1, 2, 3), must_have_shape=1)
    match = 'Array has shape () which is not allowed. Shape must be one of [3, (-1, 3)].'
    with pytest.raises(ValueError, match=escape(match)):
        validate_arrayNx3(0)
    with pytest.raises(ValueError, match='_input'):
        validate_arrayNx3([1, 2, 3, 4], name='_input')


@pytest.mark.parametrize('reshape', [True, False])
def test_validate_arrayN(reshape):  # noqa: N802
    # test 0D input is reshaped to 1D by default
    arr = validate_arrayN(0)
    assert arr.shape == (1,)
    assert np.array_equal(arr, [0])

    # test 2D input is reshaped to 1D by default
    arr = validate_arrayN([[1, 2, 3]])
    assert arr.shape == (3,)
    assert np.array_equal(arr, [1, 2, 3])

    if not reshape:
        match = 'Array has shape () which is not allowed. Shape must be -1.'
        with pytest.raises(ValueError, match=escape(match)):
            validate_arrayN(0, reshape=False)

        match = 'Array has shape (1, 3) which is not allowed. Shape must be -1.'
        with pytest.raises(ValueError, match=escape(match)):
            validate_arrayN([[1, 2, 3]], reshape=False)

    arr = validate_arrayN((1, 2, 3, 4, 5, 6), reshape=reshape)
    assert arr.shape == (6,)

    match = (
        "Parameter 'must_have_shape' cannot be set for function `validate_arrayN`.\n"
        'Its value is automatically set to `[(), -1, (1, -1)]`.'
    )
    with pytest.raises(ValueError, match=escape(match)):
        validate_arrayN((1, 2, 3), must_have_shape=1)

    match = 'Array has shape (2, 2) which is not allowed. Shape must be one of [(), -1, (1, -1)].'
    with pytest.raises(ValueError, match=escape(match)):
        validate_arrayN(((1, 2), (3, 4)))
    with pytest.raises(ValueError, match='_input'):
        validate_arrayN(((1, 2), (3, 4)), name='_input')


def test_validate_arrayN_unsigned():  # noqa: N802
    # test 0D input is reshaped to 1D by default
    arr = validate_arrayN_unsigned(0.0)
    assert arr.shape == (1,)
    assert np.array_equal(arr, [0])
    assert arr.dtype.type is np.int32 or arr.dtype.type is np.int64

    arr = validate_arrayN_unsigned(0.0, dtype_out='uint8')
    assert arr.dtype.type is np.uint8

    with pytest.raises(ValueError, match='Shape must be -1.'):
        validate_arrayN_unsigned(0.0, reshape=False)

    match = '_input values must all be greater than or equal to 0.'
    with pytest.raises(ValueError, match=match):
        validate_arrayN_unsigned([-1, 1], name='_input')


@pytest.mark.parametrize('reshape', [True, False])
def test_validate_array3(reshape):
    # test 0D input is reshaped to len-3 1D vector with broadcasting enabled
    arr = validate_array3(0, broadcast=True)
    assert arr.shape == (3,)
    assert np.array_equal(arr, [0, 0, 0])

    # test 2D input is reshaped to 1D by default
    arr = validate_array3([[1, 2, 3]])
    assert arr.shape == (3,)
    assert np.array_equal(arr, [1, 2, 3])

    arr = validate_array3([[1], [2], [3]])
    assert arr.shape == (3,)
    assert np.array_equal(arr, [1, 2, 3])

    if not reshape:
        # test check fails with 2D input and no reshape
        match = 'Array has shape (1, 3) which is not allowed. Shape must be (3,).'
        with pytest.raises(ValueError, match=escape(match)):
            validate_array3([[1, 2, 3]], reshape=reshape)

        # test correct shape with broadcast and no reshape
        match = 'Shape must be one of [(3,), (), (1,)].'
        with pytest.raises(ValueError, match=escape(match)):
            validate_array3((1, 2, 3, 4, 5, 6), reshape=reshape, broadcast=True)
    else:
        # test error match shows correct shape with broadcast and with reshape
        match = 'Shape must be one of [(3,), (1, 3), (3, 1), (), (1,)]'
        with pytest.raises(ValueError, match=escape(match)):
            validate_array3((1, 2, 3, 4, 5, 6), reshape=reshape, broadcast=True)

    # test shape cannot be overridden
    match = (
        "Parameter 'must_have_shape' cannot be set for function `validate_array3`.\n"
        'Its value is automatically set to `[(3,), (1, 3), (3, 1)]`.'
    )
    with pytest.raises(ValueError, match=escape(match)):
        validate_array3((1, 2, 3), must_have_shape=3)


def test_check_range():
    check_range((1, 2, 3), [1, 3])

    match = 'Array values must all be less than or equal to 2.'
    with pytest.raises(ValueError, match=match):
        check_range((1, 2, 3), [1, 2])

    match = 'Input values must all be greater than or equal to 2.'
    with pytest.raises(ValueError, match=match):
        check_range((1, 2, 3), [2, 3], name='Input')

    # Test strict bounds
    match = 'Array values must all be less than 3.'
    with pytest.raises(ValueError, match=match):
        check_range((1, 2, 3), [1, 3], strict_upper=True)

    match = 'Array values must all be greater than 1.'
    with pytest.raises(ValueError, match=match):
        check_range((1, 2, 3), [1, 3], strict_lower=True)


class Case(NamedTuple):
    kwarg: dict
    valid_array: np.ndarray
    invalid_array: np.ndarray
    error_type: type
    error_match: str


def numeric_array_test_cases():
    return (
        Case(
            dict(
                must_be_finite=True,
                must_be_real=False,
            ),  # must be real is only added for extra coverage
            0,
            np.inf,
            ValueError,
            'must have finite values',
        ),
        Case(dict(must_be_real=True), 0, 1 + 1j, TypeError, 'must have real numbers'),
        Case(
            dict(must_be_integer=True),
            0.0,
            0.1,
            ValueError,
            'must have integer-like values',
        ),
        Case(dict(must_be_sorted=True), [0, 1], [1, 0], ValueError, 'must be sorted'),
        Case(
            dict(must_be_sorted=dict(ascending=True, strict=False, axis=-1)),
            [0, 1],
            [1, 0],
            ValueError,
            'must be sorted',
        ),
    )


@pytest.mark.parametrize('name', ['_array', '_input'])
@pytest.mark.parametrize('copy', [True, False])
@pytest.mark.parametrize('as_any', [True, False])
@pytest.mark.parametrize('to_list', [True, False])
@pytest.mark.parametrize('to_tuple', [True, False])
@pytest.mark.parametrize('dtype_out', [np.float32, np.float64])
@pytest.mark.parametrize('case', numeric_array_test_cases())
@pytest.mark.parametrize('stack_input', [True, False])
@pytest.mark.parametrize('input_type', [tuple, list, np.ndarray, pyvista_ndarray])
def test_validate_array(
    name,
    copy,
    as_any,
    to_list,
    to_tuple,
    dtype_out,
    case,
    stack_input,
    input_type,
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
        valid_array = _cast_to_tuple(valid_array)
        invalid_array = _cast_to_tuple(invalid_array)
    elif input_type is list:
        valid_array = valid_array.tolist()
        invalid_array = invalid_array.tolist()
    elif input_type is np.ndarray:
        valid_array = np.asarray(valid_array)
        invalid_array = np.asarray(invalid_array)
    else:  # pyvista_ndarray:
        valid_array = pyvista_ndarray(valid_array)
        invalid_array = pyvista_ndarray(invalid_array)

    shape = np.array(valid_array).shape
    common_kwargs = dict(
        **case.kwarg,
        name=name,
        copy=copy,
        as_any=as_any,
        to_list=to_list,
        to_tuple=to_tuple,
        must_have_dtype=np.number,
        dtype_out=dtype_out,
        must_have_length=range(np.array(valid_array).size + 1),
        must_have_min_length=1,
        must_have_max_length=np.array(valid_array).size,
        must_have_shape=shape,
        must_have_ndim=len(shape),
        reshape_to=shape,
        broadcast_to=shape,
        must_be_in_range=(np.min(valid_array), np.max(valid_array)),
        must_be_nonnegative=np.all(np.array(valid_array) > 0),
    )

    # Test raises correct error with invalid input
    with pytest.raises(case.error_type, match=case.error_match):
        validate_array(invalid_array, **common_kwargs)
    # Test error has correct name
    with pytest.raises(case.error_type, match=name):
        validate_array(invalid_array, **common_kwargs)

    # Test no error with valid input
    array_in = valid_array
    array_out = validate_array(array_in, **common_kwargs)
    assert np.array_equal(array_out, array_in)

    # Check output
    if np.array(array_in).ndim == 0 and (to_tuple or to_list):
        # test scalar input results in scalar output
        assert isinstance(array_out, (float, int))
    elif to_tuple:
        assert type(array_out) is tuple
    elif to_list:
        assert isinstance(array_out, list)
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
                and np.dtype(dtype_out) is array_in.dtype
            ):
                assert array_out is array_in
            else:
                assert array_out is not array_in
        else:
            assert type(array_out) is np.ndarray

    if copy:
        assert array_out is not array_in


@pytest.mark.parametrize('array', [(True,), 'abc'])
def test_validate_array_non_numeric(array):
    match = 'Array must have real numbers.'
    with pytest.raises(TypeError, match=match):
        assert validate_array(array)
    assert validate_array(array, must_be_real=False)


@pytest.mark.parametrize('obj', [0, 0.0, '0'])
@pytest.mark.parametrize('classinfo', [int, (int, float), [int, float]])
@pytest.mark.parametrize('allow_subclass', [True, False])
@pytest.mark.parametrize('name', ['_input', '_object'])
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

    elif type(classinfo) is tuple:
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
    elif type(obj) is classinfo:
        check_type(obj, classinfo)
    else:
        with pytest.raises(TypeError, match=f'{name} must have type'):
            check_type(obj, classinfo, name=name)
        with pytest.raises(TypeError, match='Object must have type'):
            check_type(obj, classinfo)

    match = "Name must be a string, got <class 'int'> instead."
    with pytest.raises(TypeError, match=match):
        check_instance(0, int, name=0)


def test_check_type():
    check_type(0, int, name='abc')
    check_type(0, int)
    with pytest.raises(TypeError):
        check_type('str', int)
    with pytest.raises(TypeError):
        check_type(0, int, name=1)
    check_type(0, int | float)


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason='Union type input requires python3.10 or higher',
)
def test_check_type_union():
    check_type(0, int | float)


def test_check_string():
    check_string('abc')
    check_string('abc', name='123')
    match = "Value must be an instance of <class 'str'>. Got <class 'int'> instead."
    with pytest.raises(TypeError, match=match):
        check_string(0, name='Value')
    match = "Object must be an instance of <class 'str'>. Got <class 'int'> instead."
    with pytest.raises(TypeError, match=match):
        check_string(0)
    match = "Name must be a string, got <class 'float'> instead."
    with pytest.raises(TypeError, match=match):
        check_string('abc', name=0.0)

    class StrSubclass(str):
        pass

    check_string(StrSubclass(), allow_subclass=True)
    with pytest.raises(TypeError, match="Object must have type <class 'str'>."):
        check_string(StrSubclass(), allow_subclass=False)


def test_check_less_than():
    check_less_than([0], 1)
    check_less_than(np.eye(3), 1, strict=False)
    match = 'Array values must all be less than 0.'
    with pytest.raises(ValueError, match=match):
        check_less_than(0, 0, strict=True)
    match = '_input values must all be less than or equal to 0.'
    with pytest.raises(ValueError, match=match):
        check_less_than(1, 0, strict=False, name='_input')


def test_check_greater_than():
    check_greater_than([1], 0)
    check_greater_than(np.eye(3), 0, strict=False)
    match = 'Array values must all be greater than 0.'
    with pytest.raises(ValueError, match=match):
        check_greater_than(0, 0, strict=True)
    match = '_input values must all be greater than or equal to 0.'
    with pytest.raises(ValueError, match=match):
        check_greater_than(-1, 0, strict=False, name='_input')


def test_check_real():
    check_real(1)
    check_real(-2.0)
    check_real(np.array(2.0, dtype='uint8'))
    match = 'Array must have real numbers.'
    with pytest.raises(TypeError, match=match):
        check_real(1 + 1j)
    with pytest.raises(TypeError, match=match):
        check_real(True)
    match = '_input must have real numbers.'
    with pytest.raises(TypeError, match=match):
        check_real(1 + 1j, name='_input')


def test_check_finite():
    check_finite(0)
    match = '_input must have finite values.'
    with pytest.raises(ValueError, match=match):
        check_finite(np.nan, name='_input')


def test_check_integer():
    check_integer(1)
    check_integer([2, 3.0])
    match = (
        "Input has incorrect dtype of 'float64'. The dtype must be a "
        "subtype of <class 'numpy.integer'>."
    )
    with pytest.raises(TypeError, match=match):
        check_integer([2, 3.0], strict=True, name='_input')
    match = '_input must have integer-like values.'
    with pytest.raises(ValueError, match=match):
        check_integer([2, 3.4], strict=False, name='_input')


def test_check_sequence():
    check_sequence((1,), name='abc')
    check_sequence(range(3))
    check_sequence('abc')
    with pytest.raises(TypeError, match='_input'):
        check_sequence(np.array(1), name='_input')


def test_check_iterable():
    check_iterable((1,), name='abc')
    check_iterable(range(3))
    check_iterable('abc')
    check_iterable(np.array(1))
    with pytest.raises(TypeError, match='_input'):
        check_iterable(1, name='_input')


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
        check_length((1,), exact_length=(1, 2.4), name='_input')

    match = '_input must have a length equal to any of: 1. Got length 2 instead.'
    with pytest.raises(ValueError, match=match):
        check_length((1, 2), exact_length=1, name='_input')
    match = '_input must have a length equal to any of: [3, 4]. Got length 2 instead.'
    with pytest.raises(ValueError, match=escape(match)):
        check_length((1, 2), exact_length=[3, 4], name='_input')

    match = '_input must have a maximum length of 1. Got length 2 instead.'
    with pytest.raises(ValueError, match=match):
        check_length((1, 2), max_length=1, name='_input')

    match = '_input must have a minimum length of 2. Got length 1 instead.'
    with pytest.raises(ValueError, match=match):
        check_length((1,), min_length=2, name='_input')

    match = 'Range with 2 elements must be sorted in ascending order. Got:\n    array([4, 2])'
    with pytest.raises(ValueError, match=escape(match)):
        check_length(
            (
                1,
                2,
                3,
            ),
            min_length=4,
            max_length=2,
        )

    match = 'Shape must be -1.'
    with pytest.raises(ValueError, match=escape(match)):
        check_length(((1, 2), (3, 4)), must_be_1d=True)


def test_check_nonnegative():
    check_nonnegative(0)
    check_nonnegative(np.eye(3))
    match = 'Array values must all be greater than or equal to 0.'
    with pytest.raises(ValueError, match=match):
        check_nonnegative(-1)


@pytest.mark.parametrize('shape', [(), (8,), (4, 6), (2, 3, 4)])
@pytest.mark.parametrize('axis', [None, -1, -2, -3, 0, 1, 2, 3])
@pytest.mark.parametrize('ascending', [True, False])
@pytest.mark.parametrize('strict', [True, False])
def test_check_sorted(shape, axis, ascending, strict):
    def _check_sorted_params(arr):
        check_sorted(arr, axis=axis, strict=strict, ascending=ascending)

    if shape == ():
        # test always succeeds with scalar
        _check_sorted_params(0)
        return

    # Create ascending array with unique values
    num_elements = np.prod(shape)
    arr_strict_ascending = np.arange(num_elements).reshape(shape)

    try:
        # Create ascending array with duplicate values
        arr_ascending = np.repeat(arr_strict_ascending, 2, axis=axis)
        # Create descending arrays
        arr_descending = np.flip(arr_ascending, axis=axis)
        arr_strict_descending = np.flip(arr_strict_ascending, axis=axis)
    except np.exceptions.AxisError:
        # test ValueError is raised whenever an AxisError would otherwise be raised
        with pytest.raises(
            ValueError,
            match=f'Axis {axis} is out of bounds for ndim {arr_strict_ascending.ndim}',
        ):
            _check_sorted_params(arr_strict_ascending)
        return

    if axis is None and arr_ascending.ndim > 1:
        # test that axis=None will flatten array and cause it not to be sorted
        # for higher dimension arrays
        with pytest.raises(ValueError):  # noqa: PT011
            _check_sorted_params(arr_ascending)
        return

    if strict and ascending:
        _check_sorted_params(arr_strict_ascending)
        for a in [arr_ascending, arr_descending, arr_strict_descending]:
            with pytest.raises(ValueError, match='must be sorted in strict ascending order. Got:'):
                _check_sorted_params(a)

    elif not strict and ascending:
        _check_sorted_params(arr_ascending)
        _check_sorted_params(arr_strict_ascending)
        for a in [arr_descending, arr_strict_descending]:
            with pytest.raises(ValueError, match='must be sorted in ascending order. Got:'):
                _check_sorted_params(a)

    elif strict and not ascending:
        _check_sorted_params(arr_strict_descending)
        for a in [arr_ascending, arr_strict_ascending, arr_descending]:
            with pytest.raises(
                ValueError, match='must be sorted in strict descending order. Got:'
            ):
                _check_sorted_params(a)

    elif not strict and not ascending:
        _check_sorted_params(arr_descending)
        _check_sorted_params(arr_strict_descending)
        for a in [arr_ascending, arr_strict_ascending]:
            with pytest.raises(ValueError, match='must be sorted in descending order'):
                _check_sorted_params(a)


def test_check_iterable_items():
    check_iterable_items([1, 2, 3], int)
    check_iterable_items(('a', 'b', 'c'), str)
    check_iterable_items('abc', str)
    check_iterable_items(range(10), int)
    match = (
        "All items of Iterable must be an instance of <class 'str'>. Got <class 'int'> instead."
    )
    with pytest.raises(TypeError, match=escape(match)):
        check_iterable_items(['abc', 1], str)
    with pytest.raises(TypeError, match='All items of _input'):
        check_iterable_items(['abc', 1], str, name='_input')


def test_check_number():
    check_number(1)
    check_number(1 + 1j)
    match = (
        "_input must be an instance of <class 'numbers.Number'>. "
        "Got <class 'numpy.ndarray'> instead."
    )
    with pytest.raises(TypeError, match=match):
        check_number(np.array(0), name='_input')
    match = 'Object must be'
    with pytest.raises(TypeError, match=match):
        check_number(np.array(0))


def test_check_contains():
    check_contains(['foo', 'bar'], must_contain='foo')
    match = "Input 'foo' is not valid. Input must be one of: \n\t['cat', 'bar']"
    with pytest.raises(ValueError, match=escape(match)):
        check_contains(['cat', 'bar'], must_contain='foo')
    match = "_input '5' is not valid. _input must be in: \n\trange(0, 4)"
    with pytest.raises(ValueError, match=escape(match)):
        check_contains(range(4), must_contain=5, name='_input')


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

    # test bad input
    with pytest.raises(ValueError, match=f'{name} cannot be parallel.'):
        validate_axes([[1, 0, 0], [1, 0, 0], [0, 1, 0]], name=name)
    with pytest.raises(ValueError, match='Axes cannot be parallel.'):
        validate_axes([[0, 1, 0], [1, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError, match=f'{name} cannot be zeros.'):
        validate_axes([[1, 0, 0], [0, 1, 0], [0, 0, 0]], name=name)
    with pytest.raises(ValueError, match='Axes cannot be zeros.'):
        validate_axes([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match='Axes cannot be zeros.'):
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
    with pytest.raises(ValueError, match=f'{name} do not have a right-handed orientation.'):
        validate_axes(axes_left, must_have_orientation='right', name=name)

    validate_axes(axes_right, must_have_orientation=None)
    validate_axes(axes_right, must_have_orientation='right')
    with pytest.raises(ValueError, match=f'{name} do not have a left-handed orientation.'):
        validate_axes(axes_right, must_have_orientation='left', name=name)

    # test specifying two vectors without orientation raises error (3rd cannot be computed)
    with pytest.raises(
        ValueError,
        match=f'{name} orientation must be specified when only two vectors are given.',
    ):
        validate_axes([1, 0, 0], [0, 1, 0], must_have_orientation=None, name=name)


@pytest.mark.parametrize('bias_index', [(0, 1), (1, 0), (2, 0)])
def test_validate_axes_orthogonal(bias_index):
    axes_right = np.eye(3)
    axes_right[bias_index[0], bias_index[1]] = 0.1
    axes_left = np.array([[1, 0.0, 0], [0, 1, 0], [0, 0, -1]])
    axes_left[bias_index[0], bias_index[1]] = 0.1

    match = 'Axes are not orthogonal.'
    axes = validate_axes(
        axes_right,
        must_be_orthogonal=False,
        normalize=False,
        must_have_orientation='right',
    )
    assert np.array_equal(axes, axes_right)
    with pytest.raises(ValueError, match=match):
        validate_axes(axes_right, must_be_orthogonal=True)

    axes = validate_axes(
        axes_left,
        must_be_orthogonal=False,
        normalize=False,
        must_have_orientation='left',
    )
    assert np.array_equal(axes, axes_left)
    with pytest.raises(ValueError, match=match):
        validate_axes(axes_left, must_be_orthogonal=True)


def test_validate_rotation():
    I3 = np.eye(3)
    validated = validate_rotation(I3)
    assert np.array_equal(validated, I3)
    validated = validate_rotation(I3, must_have_handedness='right')
    assert np.array_equal(validated, I3)
    match = (
        'Rotation has incorrect handedness. Expected a left-handed rotation, '
        'but got a right-handed rotation instead.'
    )
    with pytest.raises(ValueError, match=match):
        validate_rotation(I3, must_have_handedness='left')

    validated = validate_rotation(-I3)
    assert np.array_equal(validated, -I3)
    validated = validate_rotation(-I3, must_have_handedness='left')
    assert np.array_equal(validated, -I3)
    match = (
        'Rotation has incorrect handedness. Expected a right-handed rotation, '
        'but got a left-handed rotation instead.'
    )
    with pytest.raises(ValueError, match=match):
        validate_rotation(-I3, must_have_handedness='right')

    match = 'Rotation is not valid. Rotation must be orthogonal.'
    with pytest.raises(ValueError, match=match):
        validate_rotation(I3 * 2)


def test_validate_rotation_tolerance():
    # Define valid rotation matrix which fails the check if the tolerance is too low
    # Matrix values come directly from a CI test failure
    # See https://github.com/pyvista/pyvista/pull/7053#issuecomment-2571663768
    rotation = np.array(
        [
            [6.1753786e-01, 4.8325321e-01, -6.2057501e-01],
            [-2.1952267e-04, 7.8909826e-01, 6.1426693e-01],
            [7.8654110e-01, -3.7919688e-01, 4.8740414e-01],
        ]
    )
    validate_rotation(rotation)


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


@pytest.mark.filterwarnings('ignore:Creating an ndarray from ragged nested sequences:UserWarning')
def test_cast_to_numpy_raises():
    if NUMPY_VERSION_INFO < (1, 26) and sys.platform == 'linux':
        err = TypeError
        match = 'Object arrays are not supported.'
    else:
        err = ValueError
        match = "Input cannot be cast as <class 'numpy.ndarray'>."
    with pytest.raises(err, match=match):
        _cast_to_numpy([[1], [2, 3]])

    match = 'Object arrays are not supported.'
    with pytest.raises(TypeError, match=match):
        _cast_to_numpy(list)


def test_cast_to_numpy_must_be_real():
    _ = _cast_to_numpy([0, 1], must_be_real=True)
    _ = _cast_to_numpy('abc', must_be_real=False)

    match = "Array must have real numbers. Got dtype <class 'numpy.complex128'>"
    with pytest.raises(TypeError, match=match):
        _ = _cast_to_numpy([0, 1 + 1j], must_be_real=True)
    match = "Array must have real numbers. Got dtype <class 'numpy.str_'>"
    with pytest.raises(TypeError, match=match):
        _ = _cast_to_numpy('abc', must_be_real=True)


def test_cast_to_tuple():
    array_in = np.zeros(shape=(2, 2, 3))
    array_tuple = _cast_to_tuple(array_in)
    assert array_tuple == (
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )
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


@pytest.mark.parametrize(
    ('dimensionality', 'reshape', 'expected_dimensionality'),
    [
        (0, True, 0),
        (0, False, 0),
        (1, True, 1),
        (2, True, 2),
        (3, True, 3),
        ('0D', True, 0),
        ('1D', True, 1),
        ('2D', True, 2),
        ('3D', True, 3),
        (('1D',), True, 1),
    ],
)
def test_validate_dimensionality(dimensionality, reshape, expected_dimensionality):
    assert validate_dimensionality(dimensionality, reshape=reshape) == expected_dimensionality


@pytest.mark.parametrize(
    ('dimensionality', 'message'),
    [
        (-1, 'Dimensionality values must all be greater than or equal to 0.'),
        ('5D', 'Dimensionality values must all be less than or equal to 3.'),
        (
            [1, 1],
            'Dimensionality has shape (2,) which is not allowed. Shape must be one of [(), (1,)].',
        ),
        (
            'invalid',
            '`invalid` is not a valid dimensionality. '
            'Use one of [0, 1, 2, 3, "0D", "1D", "2D", "3D"].',
        ),
    ],
)
def test_validate_dimensionality_errors(dimensionality, message):
    with pytest.raises(ValueError, match=escape(message)):
        validate_dimensionality(dimensionality)


@pytest.mark.parametrize(
    ('n_colors', 'match'),
    [
        (
            None,
            'Input must be a single ColorLike color or a sequence of ColorLike colors.',
        ),
        (
            42,
            'Input must be a single ColorLike color or a sequence of 42 ColorLike colors.',
        ),
    ],
)
def test_validate_color_sequence_raises(n_colors, match):
    with pytest.raises(ValueError, match=match):
        _validate_color_sequence('foo', n_colors=n_colors)
