from functools import wraps
from typing import Any, Union, Tuple

import numpy as np
from vtkmodules.vtkCommonMath import vtkMatrix4x4, vtkMatrix3x3
from vtkmodules.vtkCommonTransforms import vtkTransform

from pyvista import array_from_vtkmatrix
from pyvista.core.input_validation.check import check_is_subdtype, check_is_real, check_has_shape, check_is_finite, \
    check_is_integerlike, check_is_in_range, check_is_sorted, \
    check_is_dtypelike, check_is_string, \
    check_is_greater_than
from pyvista.core.input_validation.cast import cast_to_tuple_array, \
    cast_to_ndarray


def validate_numeric_array(
    arr,
    /,
    *,
    shape=None,
    dtype_base=None,
    dtype_out=None,
    name="Array",
    as_any=True,
    copy=False,
    must_be_finite=False,
    must_be_real=True,
    must_be_integer_like=False,
    must_be_sorted=False,
    must_be_in_range=None,
    strict_lower_bound=False,
    strict_upper_bound=False,
    to_list=False,
    to_tuple=False,
) -> Any:
    """Check and validate a numeric array meets specific requirements.

    Validate an array to ensure it is numeric, has a specific shape, has
    a particular data-type, and/or has values that meet specific
    requirements such as being sorted, integer-like, or finite.

    Parameters
    ----------
    arr : array_like
        Array to be validated, in any form that can be converted to
        a NumPy ndarray. This includes lists, lists of tuples, tuples,
        tuples of tuples, tuples of lists and ndarrays.

    shape : int, Tuple[int,...] | List[int, Tuple[int,...]], optional
        Check the array has a specific shape. Specify a single shape or
        a list of any allowable shapes. If an integer, the array must
        be 1-dimensional with that length. Use a value of -1 for any
        dimension where its size is allowed to vary. Use ``()`` to allow
        scalar values (i.e. 0-dimensional). Set to ``None`` if the array
        can have any shape (default).

    dtype_base : dtype-like | Iterable[dtype-like], optional
        Check the array's data-type. Specify a NumPy dtype or dtype-like
        base class which the array's data must be a subtype of. If
        iterable, the array's data must be a subtype of at least one of
        specified dtypes.

    dtype_out : dtype-like
        The desired data-type to cast the output data as.

    as_any :
        Allow subclasses of ``np.ndarray`` to pass through without
        making a copy.

    copy : bool, default: False
        If ``True``, a copy of the array is returned. A copy is always
        returned if the array:

            * is a nested sequence
            * is a subclass of ``np.ndarray`` and ``as_any`` is ``False``.

        A copy may also be made to satisfy ``dtype_out`` requirements.

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

    to_list : bool, False
        Return the validated array as a list or nested list. Scalar
        values are always returned as a number. Has no effect if
        ``to_tuple=True``.

    to_tuple : bool, False
        Return the validated array as a tuple or nested tuple. Scalar
        values are always returned as a number.

    Returns
    -------
    array_like
        Validated array. Returned object is:
            * an instance of ``np.ndarray`` (default), or
            * a nested list (if ``to_list=True``), or
            * nested tuples (if ``to_tuple=True``), or
            * a number if the input is scalar.

    """
    arr_out = cast_to_ndarray(arr, as_any=as_any, copy=copy, name=name)

    # Check type
    try:
        check_is_subdtype(arr_out, np.number, name=name)
    except TypeError as e:
        raise TypeError(f"{name} must be numeric.") from e
    if must_be_real:
        check_is_real(arr_out, name=name)
    if dtype_base is not None:
        check_is_subdtype(arr_out, dtype_base, name=name)

    if shape is not None:
        check_has_shape(arr_out, shape, name=name)

    # Check values
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
            valid_arr = validate_numeric_array(transformlike, shape=[(3, 3), (4, 4)])
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
    if isinstance(transformlike, vtkMatrix3x3):
        transformlike = array_from_vtkmatrix(transformlike)
    elif isinstance(transformlike, np.ndarray) and not transformlike.shape == (3, 3):
        raise ValueError('Transformation array must be 3x3.')
    else:
        raise TypeError('Input transform must be one of:\n' '\tvtkMatrix3x3\n' '\t3x3 np.ndarray\n')
    return transformlike


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


@wraps(validate_numeric_array)
def validate_number(num, **kwargs):
    kwargs.setdefault('name', 'Number')
    kwargs.setdefault('to_list', True)
    _set_default_kwarg_mandatory(kwargs, 'shape', (), name="Number")
    _set_default_kwarg_mandatory(kwargs, 'must_be_real', True)
    return validate_numeric_array(num, **kwargs)


def validate_data_range(rng, **kwargs):
    """Validate a data range."""
    name = 'Data Range'
    kwargs.setdefault('name', name)
    _set_default_kwarg_mandatory(kwargs, 'shape', 2, name=name)
    _set_default_kwarg_mandatory(kwargs, 'must_be_sorted', True, name=name)
    if 'to_list' not in kwargs:
        kwargs.setdefault('to_tuple', True)
    return validate_numeric_array(rng, **kwargs)


def validate_arrayNx3(arr, reshape=True, **kwargs):
    """Validate an array is numeric and has shape Nx3.

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
        Validated array with shape Nx3.

    """
    name = 'Array'
    kwargs.setdefault('name', name)

    if reshape:
        shape = [3, (-1, 3)]
    else:
        shape = (-1, 3)

    _set_default_kwarg_mandatory(kwargs, 'shape', shape, name=name)
    arr = validate_numeric_array(arr, **kwargs)
    if reshape:
        return arr.reshape((-1, 3))
    return arr


def validate_numeric_array_shape3():
    pass


def coerce_number_or_array3_as_array3():
    """Check that a sequence's elements are all strings."""
    pass


def _set_default_kwarg_mandatory(kwargs: dict, key: str, default: Any, *, name="Array"):
    """Set a kwarg and raise ValueError if not set to its default value."""
    val = kwargs.pop(key, default)
    if val != default:
        msg = (
            f"Parameter '{key}' cannot be set for {name}. Its value is "
            f"automatically set to `{default}`."
        )
        raise ValueError(msg)
    kwargs[key] = default
