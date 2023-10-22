"""Functions that check the type and/or value of inputs.

A `check_` function typically:
    * performs a simple validation on a single input variable
    * raises an error if the check fails due to invalid input
    * does not modify input or return anything

"""
from collections.abc import Iterable, Sequence
from numbers import Number
from typing import Tuple, Union, get_args, get_origin

import numpy as np

from pyvista.core.utilities.arrays import cast_to_ndarray

ShapeLike = Union[int, Tuple[int, ...], Tuple[None]]


def check_is_subdtype(arg1, arg2, /, *, name='Input'):
    """Check if a dtype is a subtype of another dtype(s).

    Parameters
    ----------
    arg1 : dtype_like | array_like
        ``dtype`` or object coercible to one. If array_like, the dtype
        of the array is used.

    arg2 : dtype_like | List[dtype_like]
        ``dtype``-like object or a list of ``dtype``-like objects.
        If a list, ``arg1`` must be a subtype of at least one of the
        specified dtypes.

    name : str, optional
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    TypeError
        If ``arg1`` is not a subtype of ``arg2``.

    """
    try:
        arg1 = cast_to_ndarray(arg1).dtype
    except ValueError:
        check_is_dtypelike(arg1)

    if not isinstance(arg2, (list, tuple)):
        arg2 = [arg2]
    valid = False
    for d in arg2:
        check_is_dtypelike(d)
        if np.issubdtype(arg1, d):
            valid = True
            break
    if not valid:
        msg = f"{name} has incorrect dtype of '{arg1}'. "
        if len(arg2) == 1:
            msg += f"The dtype must be a subtype of {arg2[0]}."
        else:
            msg += f"The dtype must be a subtype of at least one of \n{arg2}."
        raise TypeError(msg)
    return


def check_is_dtypelike(dtype):
    """Check if an input is dtype-like.

    Parameters
    ----------
    dtype : dtype_like
        ``dtype`` or object coercible to one.

    Raises
    ------
    TypeError
        If input is not coercible to a dtype object.

    """
    try:
        np.dtype(dtype)
    except TypeError as e:
        raise TypeError(f"'{dtype}' is not a valid NumPy data type.") from e


def check_is_arraylike(arr):
    """Check if an input can be cast as a NumPy ndarray.

    Parameters
    ----------
    arr : array_like
        Array to check.

    Raises
    ------
    ValueError
        If input cannot be cast as a NumPy ndarray.

    """
    try:
        cast_to_ndarray(arr)
    except ValueError as e:
        raise ValueError(*e.args) from e


def check_is_real(arr, /, *, name="Array"):
    """Check if an array has real numbers (float or integer type).

    Notes
    -----
    Arrays with NaN values are considered real and will not raise
    an error.

    Parameters
    ----------
    arr : array_like
        Array to check.

    name : str, optional
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    TypeError
        If the array does not have real numbers.

    """
    arr = cast_to_ndarray(arr)
    # Do not use np.isreal as it will fail in some cases (e.g. scalars).
    # Check dtype directly instead
    try:
        check_is_subdtype(arr, (np.floating, np.integer), name=name)
    except TypeError as e:
        raise TypeError(f"{name} must have real numbers.") from e


def check_is_sorted(arr, /, *, name="Array"):
    """Check if an array's values are sorted in ascending order.

    Parameters
    ----------
    arr : array_like
        Array to check.

    name : str, optional
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    ValueError
        If array is not sorted in ascending order.

    """
    arr = cast_to_ndarray(arr)
    if not np.array_equal(np.sort(arr), arr):
        if arr.size <= 4:
            msg_body = f"{arr}"
        else:
            msg_body = f"with {arr.size} elements"
        raise ValueError(f"{name} {msg_body} must be sorted.")


def check_is_finite(arr, /, *, name="Array"):
    """Check if an array has finite values (i.e. no NaN or Inf values).

    Parameters
    ----------
    arr : array_like
        Array to check.

    name : str, optional
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    ValueError
        If the array has any Inf or NaN values.

    """
    arr = cast_to_ndarray(arr)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must have finite values.")


def check_is_integerlike(arr, /, *, strict=False, name="Array"):
    """Check if an array has integer or integer-like float values.

    Parameters
    ----------
    arr : array_like
        Array to check.

    strict : bool, default: False
        If ``True``, the array's data must be a subtype of ``np.integer``
        (i.e. float types are not allowed).

    name : str, optional
        Variable name to use in the error messages if any are raised.


    Raises
    ------
    ValueError
        If any element's value differs from its floor.

    TypeError
        If ``strict=True` and the array's dtype is not integral.

    """
    arr = cast_to_ndarray(arr)
    if strict:
        try:
            check_is_subdtype(arr, np.integer)
        except TypeError as e:
            raise TypeError(*e.args) from e
    elif not np.array_equal(arr, np.floor(arr)):
        raise ValueError(f"{name} must have integer-like values.")


def check_is_nonnegative(arr, /, *, name="Array"):
    """Check if an array's elements are all nonnegative.

    Parameters
    ----------
    arr : array_like
        Array to check.

    name : str, optional
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    ValueError
        If the array has any negative values.

    """
    try:
        check_is_greater_than(arr, 0, strict=False, name=name)
    except ValueError as e:
        raise ValueError(*e.args) from e


def check_is_greater_than(arr, /, value, *, strict=True, name="Array"):
    """Check if array elements are all greater than some value.

    Parameters
    ----------
    arr : array_like
        Array to check.

    value : Number
        Value which the array's elements must be greater than.

    strict : bool, default: True
        If ``True``, the array's value must be strictly greater than
        ``value``. Otherwise, values must be greater than or equal to
        ``value``.

    name : str, optional
        Variable name to use in the error messages if any are raised.



    Raises
    ------
    ValueError
        If not all array elements are greater than (or equal to if
        ``strict=True``) the specified value.

    """
    arr = cast_to_ndarray(arr)
    value = cast_to_ndarray(value)
    check_has_shape(value, ())
    check_is_real(value)
    check_is_finite(value)
    if strict and not np.all(arr > value):
        raise ValueError(f"{name} values must all be greater than {value}.")
    elif not np.all(arr >= value):
        raise ValueError(f"{name} values must all be greater than or equal to {value}.")


def check_is_less_than(arr, /, value, *, strict=True, name="Array"):
    """Check array elements are all less than some value.

    Parameters
    ----------
    arr : array_like
        Array to check.

    value : scalar
        Value which the array's elements must be less than.

    strict : bool, default: True
        If ``True``, the array's value must be strictly less than
        ``value``. Otherwise, values must be less than or equal to
        ``value``.

    name : str, optional
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    ValueError
        If not all array elements are less than (or equal to if
        ``strict=True``) the specified value.

    """
    arr = cast_to_ndarray(arr)
    if strict and not np.all(arr < value):
        raise ValueError(f"{name} values must all be less than {value}.")
    elif not np.all(arr <= value):
        raise ValueError(f"{name} values must all be less than or equal to {value}.")


def check_is_in_range(arr, /, rng, *, strict_lower=False, strict_upper=False, name="Array"):
    """Check if an array's values are all within a specific range.

    Parameters
    ----------
    arr : array_like
        Array to check.

    rng : array_like, optional
        Array-like with two elements ``[min, max]`` specifying the minimum
        and maximum data values allowed, respectively. By default, the
        range endpoints are inclusive, i.e. values must be >= min
        and <= max. Use ``strict_lower`` and/or ``strict_upper``
        to further restrict the allowable range. Use ``np.inf`` or
        ``-np.inf`` to specify open intervals, e.g. ``[0, np.inf]``.

    strict_lower : bool, default: False
        Enforce a strict lower bound for the range, i.e. array values
        must be strictly greater than the minimum.

    strict_upper : bool, default: False
        Enforce a strict upper bound for the range, i.e. array values
        must be strictly less than the maximum.

    name : str, optional
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    ValueError
        If any array value is outside the specified range.

    """
    rng = cast_to_ndarray(rng)
    check_has_shape(rng, 2, name="Range")
    check_is_real(rng, name="Range")
    check_is_sorted(rng, name="Range")

    arr = cast_to_ndarray(arr)
    try:
        check_is_greater_than(arr, rng[0], strict=strict_lower, name=name)
        check_is_less_than(arr, rng[1], strict=strict_upper, name=name)
    except ValueError as e:
        raise ValueError(*e.args) from e


def check_has_shape(
    arr,
    /,
    shape,
    *,
    name="Array",
):
    """Check if an array has the specified shape.

    Parameters
    ----------
    arr : array_like
        Array to check.

    shape : int, Tuple[int,...] | List[int, Tuple[int,...]], optional
        A single shape or a list of any allowable shapes. If an integer,
        ``i``, the shape is interpreted as ``(i,)``. Use a value of
         -1 for any dimension where its size is allowed to vary, e.g.
         ``(-1,3)`` if any Nx3 array is allowed. Use ``()`` for the
          shape of scalar values (i.e. 0-dimensional). If a list, the
          array must have at least one of the specified shapes.

    name : str, optional
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    ValueError
        If the array does not have any of the specified shape(s).

    """
    arr = cast_to_ndarray(arr)
    is_error = True

    if not isinstance(shape, list):
        shape = [shape]

    for allowable_shape in shape:
        allowable_shape = _validate_shape_value(allowable_shape)
        # Compare actual shape dims to allowable shape dims
        if len(arr.shape) != len(allowable_shape):
            is_matching = False
        else:
            is_matching = True
            for i, dim in enumerate(allowable_shape):
                if dim == -1:
                    continue
                elif dim != arr.shape[i]:
                    is_matching = False
                    break
        if is_matching:
            is_error = False
            break
    if is_error:
        msg = f"{name} has shape {arr.shape} which is not allowed. "
        if len(shape) == 1:
            msg += f"Shape must be {shape[0]}."
        else:
            msg += f"Shape must be one of {shape}."
        raise ValueError(msg)


def check_is_number(num, /, *, name='Object'):
    """Check if an object is a number.

    A number is any instance of ``numbers.Number``, e.g.  ``int``,
    ``float``, and ``complex``.

    Notes
    -----
    A NumPy ndarray is not an instance of ``Number``.

    Parameters
    ----------
    num : Number
        Number to check.

    name : str, default: "Object"
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    TypeError
        If input is not an instance of ``Number``.

    """
    try:
        check_is_instance(num, Number, allow_subclass=True, name=name)
    except TypeError as e:
        raise TypeError(*e.args) from e


def check_is_string(obj: str, /, *, allow_subclass=True, name: str = 'Object'):
    """Check if an object is a string.

    Parameters
    ----------
    obj : str
        Object to check.

    allow_subclass : bool, default: True
        If ``True``, the object's type must be ``str`` or a subclass of
        ``str``. Otherwise, subclasses are not allowed.

    name : str, default: "Object"
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    TypeError
        If input is not an instance of ``str``.

    """
    try:
        check_is_instance(obj, str, allow_subclass=allow_subclass, name=name)
    except TypeError as e:
        raise TypeError(*e.args) from e


def check_is_sequence(obj: Sequence, /, *, name: str = 'Object'):
    """Check if an object is an instance of ``Sequence``.

    Parameters
    ----------
    obj : Sequence
        Object to check.

    name : str, default: "Object"
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    TypeError
        If input is not an instance of ``Sequence``.

    """
    try:
        check_is_instance(obj, Sequence, allow_subclass=True, name=name)
    except TypeError as e:
        raise TypeError(*e.args) from e


def check_is_iterable(obj: Iterable, /, *, name: str = 'Object'):
    """Check if an object is an instance of ``Iterable``.

    Parameters
    ----------
    obj : Iterable
        Iterable object to check.

    name : str, default: "Object"
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    TypeError
        If input is not an instance of ``Iterable``.

    """
    try:
        check_is_instance(obj, Iterable, allow_subclass=True, name=name)
    except TypeError as e:
        raise TypeError(*e.args) from e


def check_is_instance(
    obj, /, classinfo: Union[type, Tuple[type, ...]], *, allow_subclass=True, name: str = 'Object'
):
    """Check that an object is an instance of the given types.

    Parameters
    ----------
    obj : Any
        Object to check.

    classinfo : type | Tuple[type, ...]
        ``type`` or tuple of types. Object must be an instance of one of
        the types.

    allow_subclass : bool, default: True
        If ``True``, the object's type must be specified by ``classinfo``
         or any of its subclasses. Otherwise, subclasses are not allowed.

    name : str, optional
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    TypeError
        If object is not an instance of any of the given types.

    """
    if not isinstance(name, str):
        raise TypeError(f"Name must be a string, got {type(name)} instead.")

    # Get class info from generics
    if get_origin(classinfo) is Union:
        classinfo = get_args(classinfo)

    # Count num classes
    if isinstance(classinfo, tuple):
        num_classes = len(classinfo)
    else:
        num_classes = 1

    # Check if is instance
    is_instance = isinstance(obj, classinfo)

    # Set flag to raise error if not instance
    is_error = False
    if allow_subclass and not is_instance:
        is_error = True
        if num_classes == 1:
            msg_body = "must be an instance of"
        else:
            msg_body = "must be an instance of any type"

    # Set flag to raise error if not type
    elif not allow_subclass:
        if isinstance(classinfo, tuple):
            if type(obj) not in classinfo:
                is_error = True
                msg_body = "must have one of the following types"
        elif type(obj) is not classinfo:
            is_error = True
            msg_body = "must have type"

    if is_error:
        msg = f"{name} {msg_body} {classinfo}. Got {type(obj)} instead."
        raise TypeError(msg)


def check_is_type(obj, /, classinfo, *, name: str = 'Object'):
    """Check that object is one of the given types.

    Parameters
    ----------
    obj : Any
        Object to check.

    classinfo : type | Tuple[type, ...]
        ``type`` or tuple of types. Object must be one of the types.

    name : str, optional
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    TypeError
        If object is not any of the given types..

    """
    try:
        check_is_instance(obj, classinfo, allow_subclass=False, name=name)
    except TypeError as e:
        raise TypeError(*e.args) from e


def check_is_iterable_of_some_type(
    iterable_obj: Iterable,
    some_type: Union[type, Tuple[type, ...]],
    /,
    *,
    allow_subclass=True,
    name: str = 'Iterable',
):
    """Check that an iterable's items all have a specified type.

    Parameters
    ----------
    iterable_obj : Iterable
        Iterable to check.

    some_type : type | Tuple[type, ...]
        Class type(s) to check for. Each element of the sequence must
        have the type or one of the types specified.

    allow_subclass : bool, default: True
        If ``True``, the type of the iterable's items must be any of the
        given types or a subclass thereof. Otherwise, subclasses are not
        allowed.

    name : str, optional
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    TypeError
        If any of the elements have an incorrect type.

    """
    check_is_iterable(iterable_obj, name=name)
    try:
        [
            check_is_instance(
                item, some_type, allow_subclass=allow_subclass, name=f"All items of {name}"
            )
            for item in iterable_obj
        ]
    except TypeError as e:
        raise TypeError(*e.args) from e


def check_is_iterable_of_strings(
    iterable_obj: Iterable, /, *, allow_subclass=True, name: str = 'String Iterable'
):
    """Check that an iterable's items are all strings.

    Parameters
    ----------
    iterable_obj : Iterable
        Iterable to check.

    allow_subclass : bool, default: True
        If ``True``, the type of the iterable's items must be any of the
        given types or a subclass thereof. Otherwise, subclasses are not
        allowed.

    name : str, optional
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    TypeError
        If any of the elements have an incorrect type.

    """
    try:
        check_is_iterable_of_some_type(iterable_obj, str, allow_subclass=allow_subclass, name=name)
    except TypeError as e:
        raise TypeError(*e.args) from e


def check_string_is_in_iterable(string_in, string_iterable, /, *, name: str = 'String'):
    """Check that a string is in an iterable of strings.

    Parameters
    ----------
    string_in : str
        String to check.

    string_iterable : Iterable[str, ...]
        Iterable containing only strings.

    name : str, default: "String"
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    ValueError
        If the string is not in the iterable.

    """
    check_is_string(string_in, name=name)
    check_is_iterable_of_strings(string_iterable)
    if string_in not in string_iterable:
        raise ValueError(
            f"{name} '{string_in}' is not in the iterable. "
            f"{name} must be one of: \n\t" + str(string_iterable)
        )


def check_length(
    arr,
    /,
    *,
    exact_length=None,
    min_length=None,
    max_length=None,
    must_be_1d=False,
    allow_scalars=False,
    name="Array",
):
    """Check the length of an array meets specific requirements.

    Notes
    -----
    By default, this function operates on multidimensional arrays,
    where ``len(arr)`` may differ from the number of elements in the
    array. For one-dimensional cases (where ``len(arr) == arr.size``),
    set ``must_be_1D=True``.

    Parameters
    ----------
    arr : array_like
        Array to check.

    exact_length : array_like
        Check that the array has the given length. If multiple
        numbers are given, the array's length must match one of the
        numbers.

    min_length : int, optional
        Check that array has this length or larger.

    max_length : int, optional
        Check that array has this length or smaller.

    must_be_1d : bool, default: False
        If ``True``, the array is also checked that it is one-dimensional.

    allow_scalars : bool, default: False
        If ``True``, a scalar input will be reshaped to have a length of
        1. Otherwise, the check will fail since a scalar does not
        have a length.

    name : str, optional
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    ValueError
        If the array's length is outside the specified range.

    """
    if allow_scalars:
        # Reshape to 1D
        if isinstance(arr, np.ndarray) and arr.ndim == 0:
            arr = [arr.tolist()]
        elif isinstance(arr, Number):
            arr = [arr]
    check_is_instance(arr, (Sequence, np.ndarray), name=name)

    if must_be_1d:
        check_has_shape(arr, shape=(-1))

    if exact_length is not None:
        exact_length = np.array(exact_length)
        check_is_integerlike(exact_length, name="'exact_length'")
        if len(arr) not in exact_length:
            raise ValueError(
                f"{name} must have a length equal to any of: {exact_length}. "
                f"Got length {len(arr)} instead."
            )

    # Validate min/max length
    if min_length is not None:
        min_length = cast_to_ndarray(min_length)
        check_is_scalar(min_length, name="Min length")
        check_is_real(min_length, name="Min length")
    if max_length is not None:
        max_length = cast_to_ndarray(max_length)
        check_is_scalar(max_length, name="Max length")
        check_is_real(max_length, name="Max length")
    if min_length is not None and max_length is not None:
        check_is_sorted((min_length, max_length), name="Range")

    if min_length is not None:
        if len(arr) < min_length:
            raise ValueError(
                f"{name} must have a minimum length of {min_length}. "
                f"Got length {len(arr)} instead."
            )
    if max_length is not None:
        if len(arr) > max_length:
            raise ValueError(
                f"{name} must have a maximum length of {max_length}. "
                f"Got length {len(arr)} instead."
            )


def _validate_shape_value(
    shape: Union[int, Tuple[int, ...], Tuple[None]]
) -> Union[Tuple[None], Tuple[int, ...]]:
    """Validate shape-like input and return its tuple representation."""
    if shape is None:
        # `None` is used to mean `any shape is allowed` by the array
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


def check_is_scalar(scalar, /, *, name="Scalar"):
    """Check if an object is a real-valued scalar.

    Parameters
    ----------
    scalar : int | float | np.ndarray
        Real number as an int, float, or 0-dimensional array.

    name : str, optional
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    TypeError
        If input is not an instance of ``Number``.

    """
    check_is_instance(scalar, (int, float, np.ndarray), name=name)
    if isinstance(scalar, np.ndarray):
        check_is_real(scalar, name=name)
        if scalar.ndim > 0:
            raise ValueError(
                f"{name} must be a 0-dimensional array, got `ndim={scalar.ndim}` instead."
            )
