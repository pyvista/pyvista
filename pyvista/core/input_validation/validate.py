"""Functions that validate input and return a standard representation.

A `validate_` function typically :
    * uses `check_` functions to validate input args/kwargs
    * accepts multiple input types with similar representations
        and standardizes the output as a single representation
    * applies (optional) constraints, e.g. input or output must have a
        specific length, shape, type, data-type, etc.

"""
from functools import wraps
import inspect
from typing import Any, Tuple, Union

import numpy as np

from pyvista.core._vtk_core import vtkMatrix3x3, vtkMatrix4x4, vtkTransform
from pyvista.core.input_validation.check import (
    check_has_shape,
    check_is_dtypelike,
    check_is_finite,
    check_is_greater_than,
    check_is_in_range,
    check_is_integerlike,
    check_is_nonnegative,
    check_is_real,
    check_is_sorted,
    check_is_string,
    check_is_subdtype,
    check_length,
)
from pyvista.core.utilities.arrays import array_from_vtkmatrix, cast_to_ndarray, cast_to_tuple_array


def validate_array(
    arr,
    /,
    *,
    shape=None,
    reshape=None,
    broadcast_to=None,
    dtype_base=None,
    dtype_out=None,
    length=None,
    min_length=None,
    max_length=None,
    must_be_nonnegative=False,
    must_be_finite=False,
    must_be_real=True,
    must_be_integer_like=False,
    must_be_sorted=False,
    must_be_in_range=None,
    strict_lower_bound=False,
    strict_upper_bound=False,
    as_any=True,
    copy=False,
    to_list=False,
    to_tuple=False,
    name="Array",
) -> Any:
    """Check and validate a numeric array meets specific requirements.

    Validate an array to ensure it is numeric, has a specific shape,
    data-type, and/or has values that meet specific
    requirements such as being sorted, integer-like, or finite.

    Parameters
    ----------
    arr : array_like
        Array to be validated, in any form that can be converted to
        a NumPy ndarray. This includes lists, lists of tuples, tuples,
        tuples of tuples, tuples of lists and ndarrays.

    shape : int | Tuple[int,...] | List[int, Tuple[int,...]], optional
        Check the array has a specific shape. Specify a single shape or
        a list of any allowable shapes. If an integer, the array must
        be 1-dimensional with that length. Use a value of -1 for any
        dimension where its size is allowed to vary. Use ``()`` to allow
        scalar values (i.e. 0-dimensional). Set to ``None`` if the array
        can have any shape (default).

    reshape : int, Tuple[int,...], optional
        Reshape the output array with :func:`numpy.reshape`. The shape
        should be compatible with the original shape. If an integer,
        then the result will be a 1-D array of that length. One shape
        dimension can be -1.

    broadcast_to : int, Tuple[int,...}, optional
        Broadcast the array to a read-only view with the specified shape.
        Broadcasting is done after reshaping (if ``reshape=True``).

    dtype_base : dtype_like | List[dtype_like], optional
        Check the array's data-type. Specify a NumPy dtype or dtype-like
        base class which the array's data must be a subtype of. If
        iterable, the array's data must be a subtype of at least one of
        specified dtypes.

    dtype_out : dtype_like, optional
        The desired data-type of the returned array.

    length : array_like
        Check that the array has the given length. If multiple
        numbers are given, the array's length must match one of the
        numbers.

        .. note ::

            The array's length is determined after reshaping the array
            (if ``reshape`` is not ``None``) and after broadcasting (if
            ``broadcast`` is not ``None``). Therefore, the values of
            `length`` should take the array's new shape into consideration.

    min_length : int, optional
        Check that the array's length is this value or larger.

    max_length : int, optional
        Check that the array' length is this value or smaller.

    must_be_non_negative : bool, default: False
        Check that all elements of the array are nonnegative.

    must_be_finite : bool, default: False
        Check that all elements of the array are finite, i.e. not
        infinity and not Not a Number (NaN).

    must_be_real : bool, default: True
        Check that the array's has real numbers, i.e. its data type is
        integer or floating.

    must_be_integer_like : bool, default: False
        Check that the array's values are integer-like (i.e. that
        ``np.all(arr, np.floor(arr))``).

    must_be_sorted : bool, default: False
        Check that the array's values are sorted in ascending order.

    must_be_in_range : array_like, optional
        Check that the array's values are all within a specific range.
        Range must be array-like with two elements specifying the minimum
        and maximum data values allowed, respectively. By default, the
        range endpoints are inclusive, i.e. values must be >= minimum
        and <= maximum. Use ``strict_lower_bound`` and/or ``strict_upper_bound``
        to further restrict the allowable range.

    strict_lower_bound : bool, False
        Enforce a strict lower bound for the range specified by
        ``must_be_in_range``, i.e. array values must be strictly greater
        than the specified minimum.

    strict_upper_bound : bool, False
        Enforce a strict upper bound for the range specified by
        ``must_be_in_range``, i.e. array values must be strictly less
        than the specified maximum.

    as_any :
        Allow subclasses of ``np.ndarray`` to pass through without
        making a copy.

    copy : bool, default: False
        If ``True``, a copy of the array is returned. A copy is always
        returned if the array:

            * is a nested sequence
            * is a subclass of ``np.ndarray`` and ``as_any`` is ``False``.

        A copy may also be made to satisfy ``dtype_out`` requirements.

    to_list : bool, False
        Return the validated array as a list or nested list. Scalar
        values are always returned as a number. Has no effect if
        ``to_tuple=True``.

    to_tuple : bool, False
        Return the validated array as a tuple or nested tuple. Scalar
        values are always returned as a number.

    name : str, optional
        Variable name to use in the error messages if any of the
        validation checks fail.

    Returns
    -------
    array_like
        Validated array. Returned object is:
            * an instance of ``np.ndarray`` (default), or
            * a nested list (if ``to_list=True``), or
            * a nested tuple (if ``to_tuple=True``), or
            * a number (scalar) if the input is a number.

    """
    arr_out = cast_to_ndarray(arr, as_any=as_any, copy=copy)

    # Check type
    try:
        check_is_subdtype(arr_out, np.number, name=name)
    except TypeError as e:
        raise TypeError(f"{name} must be numeric.") from e
    if must_be_real:
        check_is_real(arr_out, name=name)
    if dtype_base is not None:
        check_is_subdtype(arr_out, dtype_base, name=name)

    # Check shape
    if shape is not None:
        check_has_shape(arr_out, shape, name=name)

    # Do reshape _after_ checking shape to prevent unexpected reshaping
    if reshape is not None and arr_out.shape != reshape:
        arr_out = arr_out.reshape(reshape)

    if broadcast_to is not None and arr_out.shape != broadcast_to:
        arr_out = np.broadcast_to(arr_out, broadcast_to, subok=True)

    # Check length _after_ reshaping otherwise length may be wrong
    if length is not None or min_length is not None or max_length is not None:
        check_length(
            arr,
            exact_length=length,
            min_length=min_length,
            max_length=max_length,
            allow_scalars=True,
            name=name,
        )

    # Check data values
    if must_be_nonnegative:
        check_is_nonnegative(arr_out, name=name)
    if must_be_finite:
        check_is_finite(arr_out, name=name)
    if must_be_integer_like:
        check_is_integerlike(arr_out, strict=False, name=name)
    if must_be_in_range is not None:
        check_is_in_range(
            arr_out,
            must_be_in_range,
            strict_lower=strict_lower_bound,
            strict_upper=strict_upper_bound,
            name=name,
        )
    if must_be_sorted:
        check_is_sorted(arr_out, name=name)

    # Process output
    if dtype_out:
        check_is_dtypelike(dtype_out)
        # Copy was done earlier, so don't do it again here
        arr_out = arr_out.astype(dtype_out, copy=False)
    if to_tuple:
        return cast_to_tuple_array(arr_out)
    if to_list:
        return arr_out.tolist()
    return arr_out


def validate_transform_as_array4x4(transformlike, /, *, name="Transform") -> np.ndarray:
    """Validate transform-like input as a 4x4 ndarray.

    Parameters
    ----------
    transformlike : array_like | vtkTransform | vtkMatrix4x4 | vtkMatrix3x3
        Transformation matrix as a 3x3 or 4x4 array, 3x3 or 4x4 vtkMatrix,
        or as a vtkTransform.

    Returns
    -------
    np.ndarray
        Validated 4x4 transformation matrix.

    """
    check_is_string(name, name="Name")
    arr = np.eye(4)  # initialize
    if isinstance(transformlike, vtkMatrix4x4):
        arr = array_from_vtkmatrix(transformlike)
    elif isinstance(transformlike, vtkMatrix3x3):
        arr[:3, :3] = array_from_vtkmatrix(transformlike)
    elif isinstance(transformlike, vtkTransform):
        arr = array_from_vtkmatrix(transformlike.GetMatrix())
    else:
        try:
            valid_arr = validate_array(transformlike, shape=[(3, 3), (4, 4)], name=name)
            if valid_arr.shape == (3, 3):
                arr[:3, :3] = valid_arr
            else:
                arr = valid_arr
        except ValueError:
            raise TypeError(
                'Input transform must be one of:\n'
                '\tvtkMatrix4x4\n'
                '\tvtkMatrix3x3\n'
                '\tvtkTransform\n'
                '\t4x4 np.ndarray\n'
                '\t3x3 np.ndarray\n'
            )

    return arr


def validate_transform_as_array3x3(transformlike, /, *, name="Transform"):
    """Validate transform-like input as a 3x3 ndarray.

    Parameters
    ----------
    transformlike : array_like | vtkMatrix3x3
        Transformation matrix as a 3x3 array or vtkMatrix3x3.

    Returns
    -------
    np.ndarray
        3x3 array.
    """
    check_is_string(name, name="Name")
    arr = np.eye(3)  # initialize
    if isinstance(transformlike, vtkMatrix3x3):
        arr[:3, :3] = array_from_vtkmatrix(transformlike)
    else:
        try:
            arr = validate_array(transformlike, shape=(3, 3), name=name)
        except ValueError:
            raise TypeError(
                'Input transform must be one of:\n' '\tvtkMatrix3x3\n' '\t3x3 np.ndarray\n'
            )
    return arr


def validate_shape_value(
    shape: Union[int, Tuple[int, ...], Tuple[None]], /, *, name="Shape"
) -> Union[Tuple[None], Tuple[int, ...]]:
    """Validate shape-like input and return its tuple representation."""
    if shape is None:
        # `None` is used to mean 'any shape is allowed` by the array
        #  validation methods, so raise an error here.
        #  Also, setting `None` as a shape is deprecated by NumPy.
        raise TypeError("`None` is not a valid shape. Use `()` instead.")
    if shape == ():
        return ()

    # Make sure shape is scalar or 1-dimensional
    # Values must be non-zero integers, and -1 is accepted
    shape_arr = cast_to_ndarray(shape)
    if shape_arr.ndim > 1:
        raise ValueError("Shape must be scalar or 1-dimensional.")
    check_is_subdtype(shape_arr, np.integer, name="Shape")
    check_is_greater_than(shape_arr, -1, name="Shape", strict=False)

    if shape_arr.ndim == 0:
        return (int(shape_arr),)
    return tuple(shape_arr)


def validate_dtype(dtype_like, /, *, name="Data type") -> np.dtype:
    """Validate dtype-like input and return it as a dtype object.

    .. warning::

        Validating a type can result in a dtype object whose type
        differs from the specified input. E.g., the type ``np.number``
        is validated as a dtype object with type ``np.float64``, not
        ``np.number``. To only check that an input is dtype-like
        without converting it to a dtype object, use :func:`~check_is_dtypelike`
        instead.

    Parameters
    ----------
    dtype_like : dtype-like
        Data type to validate.

    Returns
    -------
    np.dtype
        Data type object.

    """
    try:
        return np.dtype(dtype_like)
    except TypeError as e:
        raise TypeError(f"'{dtype_like}' is not a valid NumPy data type.") from e


@wraps(validate_array)
def validate_number(num, **kwargs):
    """Validate a number.

    By default, the number is checked to ensure it:
        * is scalar or is an array which can be cast as a scalar
        * is a real number
        * is finite

    The returned value is an `int` or `float`.

    """
    kwargs.setdefault('name', 'Number')
    kwargs.setdefault('to_list', True)
    kwargs.setdefault('must_be_finite', True)
    kwargs.setdefault('must_be_real', True)
    _set_default_kwarg_mandatory(kwargs, 'shape', ())
    return validate_array(num, **kwargs)


def validate_data_range(rng, **kwargs):
    """Validate a data range."""
    kwargs.setdefault('name', 'Data Range')
    kwargs.setdefault('must_be_real', True)
    _set_default_kwarg_mandatory(kwargs, 'shape', 2)
    _set_default_kwarg_mandatory(kwargs, 'must_be_sorted', True)
    if 'to_list' not in kwargs:
        kwargs.setdefault('to_tuple', True)
    return validate_array(rng, **kwargs)


def validate_arrayNx3(arr, /, *, reshape=True, **kwargs):
    """Validate an array is numeric and has shape Nx3.

    The array is checked to ensure its input values:
        * have shape (N,3) or can be reshaped to (N,3)
        * are numeric

    The returned array is formatted so that its values:
        * have shape (N,3)

    Parameters
    ----------
    arr : array_like
        Array to validate.

    reshape : bool, True
        If ``True``, 1D arrays with 3 elements are considered valid input
        and are reshaped to (1,3) to ensure the output is two-dimensional.

    Returns
    -------
    np.ndarray
        Validated array with shape (N,3).

    """
    if reshape:
        shape = [3, (-1, 3)]
        _set_default_kwarg_mandatory(kwargs, 'reshape', (-1, 3))
    else:
        shape = (-1, 3)
    _set_default_kwarg_mandatory(kwargs, 'shape', shape)

    return validate_array(arr, **kwargs)


def validate_arrayN(arr, /, *, reshape=True, **kwargs):
    """Validate a numeric 1D array.

    The array is checked to ensure its input values:
        * have shape (N,) or can be reshaped to (N,)
        * are numeric

    The returned array is formatted so that its values:
        * have shape (N,)

    Parameters
    ----------
    arr : array_like
        Array to validate.

    reshape : bool, True
        If ``True``, 0-dimensional scalars are reshaped to (1,) and 2D
        vectors with shape (1, N) are reshaped to (N,) to ensure the
        output is consistently one-dimensional. Otherwise, all scalar and
        2D inputs are not considered valid.

    Returns
    -------
    np.ndarray
        Validated 1D array.

    """
    if reshape:
        shape = [(), (-1), (1, -1)]
        _set_default_kwarg_mandatory(kwargs, 'reshape', (-1))
    else:
        shape = -1
    _set_default_kwarg_mandatory(kwargs, 'shape', shape)
    return validate_array(arr, **kwargs)


def validate_uintlike_arrayN(arr, /, *, reshape=True, **kwargs):
    """Validate a numeric 1D array of non-negative integers.

    The array is checked to ensure its input values:
        * have shape (N,) or can be reshaped to (N,)
        * are integer-like
        * are non-negative

    The returned array is formatted so that its values:
        * have shape (N,)
        * have an integer data type

    Parameters
    ----------
    arr : array_like
        Array to validate.

    reshape : bool, True
        If ``True``, 0-dimensional scalars are reshaped to (1,) and 2D
        vectors with shape (1, N) are reshaped to (N,) to ensure the
        output is consistently one-dimensional. Otherwise, all scalar and
        2D inputs are not considered valid.

    Returns
    -------
    np.ndarray
        Validated 1D array with non-negative integers.

    """
    # Set default dtype out but allow overriding as long as the dtype
    # is also integral
    kwargs.setdefault('dtype_out', int)
    check_is_subdtype(kwargs['dtype_out'], np.integer)

    _set_default_kwarg_mandatory(kwargs, 'must_be_integer_like', True)
    _set_default_kwarg_mandatory(kwargs, 'must_be_nonnegative', True)

    return validate_arrayN(arr, reshape=reshape, **kwargs)


def validate_array3(arr, /, *, reshape=True, broadcast=False, **kwargs):
    """Validate a numeric 1D array with 3 elements.

    Parameters
    ----------
    arr : array_like
        Array to validate.

    reshape : bool, True
        If ``True``, 2D vectors with shape (1, 3) are considered valid
        input, and are reshaped to (3,) to ensure the output is
        consistently one-dimensional.

    broadcast : bool, False
        If ``True``, scalar values or 1D arrays with a single element
        are considered valid input and the single value is broadcast to
        a length 3 array.

    Returns
    -------
    np.ndarray
        Validated 1D array with 3 elements.

    """
    shape = [(3,)]
    if reshape:
        shape.append((1, 3))
        _set_default_kwarg_mandatory(kwargs, 'reshape', (-1))
    if broadcast:
        shape.append(())  # allow 0D scalars
        shape.append((1,))  # 1D 1-element vectors
        _set_default_kwarg_mandatory(kwargs, 'broadcast_to', (3,))
    _set_default_kwarg_mandatory(kwargs, 'shape', shape)

    return validate_array(arr, **kwargs)


def _set_default_kwarg_mandatory(kwargs: dict, key: str, default: Any):
    """Set a kwarg and raise ValueError if not set to its default value."""
    val = kwargs.pop(key, default)
    if val != default:
        calling_fname = inspect.stack()[1].function
        msg = (
            f"Parameter '{key}' cannot be set for function `{calling_fname}`.\n"
            f"Its value is automatically set to `{default}`."
        )
        raise ValueError(msg)
    kwargs[key] = default
