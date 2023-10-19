"""Internal input validation functions.

Some function naming conventions used in this module are:

    check_ :
        * performs a simple validation on a single input variable
        * raises an error if the check fails due to invalid input
        * does not modify input or return anything
        * "name" parameter is typically specified

    coerce_ :
        * accepts an input variable which may have variable type and/or
        properties
        * typically used when inputs may not be array-like or when
        array shape of the input differs from the output
        * input is coerced and possibly modified into a standard form
        * uses `check_` functions to validate all input args/kwargs
        * "name" parameter is typically specified

    validate_ :
        * uses `check_` functions to validate all input args/kwargs
        * returns a similar version of an input but with constraints applied
        * is mainly used for array-like input -> array-like output
        * input shape is the same as output shape
        * "name" parameter is typically specified
        * Examples of constraints on the output:
            * returned type may be specified (e.g. as np.array or tuple
            array or list array)
            * returned dtype may be specified (e.g. float, double)
            * values known to be integral, sorted, within some range, etc.
"""
from collections.abc import Iterable, Sequence
from numbers import Number
from typing import Tuple, Union, get_args, get_origin
from warnings import warn

import numpy as np

from pyvista.core.input_validation.cast import cast_to_ndarray

ShapeLike = Union[int, Tuple[int, ...], Tuple[None]]


def check_iterable_elements_have_type(
    obj: Iterable,
    /,
    check_type: Union[type, Tuple[type, ...]],
    *,
    allow_subclass=True,
    name='Object',
):
    """Check the type of all items in an iterable.

    Parameters
    ----------
    obj : Iterable
        Iterable to check.

    check_type : type | Tuple[type, ...]
        Class type(s) to check for. Each element of the sequence must
        have the type or one of the types specified.

    name : str, optional
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    TypeError
        If the input is not a Sequence with elements of the specified type(s).

    """
    check_is_string(name, name="Name")
    check_is_iterable(obj, name=name)
    _ = (check_is_instance(element, check_type, allow_subclass=allow_subclass) for element in obj)


def check_is_subdtype(arr, /, dtypelike, *, name='Input'):
    """Check if an array's data type is a subtype a specified dtype(s).

    Parameters
    ----------
    arr : array_like
        Array to check.

    dtypelike : dtype-like | List[dtype-like]
        Data type which the array's data must be a subtype of. If
        iterable, the array's data must be a subtupe of at least one of
        specified dtypes.

    name : str, optional
        Name to use in the error messages.

    Raises
    ------
    TypeError
        If the array data not a subtype of ``dtype``.

    Returns
    -------
    bool
        Returns ``True`` if the array data is a subtupe of ``dtype``.

    """
    check_is_string(name, name="Name")
    arr = cast_to_ndarray(arr, name=name)
    if not isinstance(dtypelike, (list, tuple)):
        dtypelike = [dtypelike]
    valid = False
    for d in dtypelike:
        check_is_dtypelike(d)
        if np.issubdtype(arr.dtype, d):
            valid = True
            break
    if not valid:
        msg = f"{name} has incorrect dtype of '{arr.dtype}'. "
        if len(dtypelike) == 1:
            msg += f"The dtype must be a subtype of {dtypelike[0]}."
        else:
            msg += f"The dtype must be a subtype of at least one of \n{dtypelike}."
        raise TypeError(msg)
    return


def check_is_dtypelike(dtypelike, /, *, name="Data type"):
    """Check that an input is dtype-like.

    Parameters
    ----------
    dtypelike : dtype-like
        DType-like value to check.

    """
    # Avoid circular import
    from pyvista.core.input_validation.validate import validate_dtype

    validate_dtype(dtypelike)


def check_is_arraylike(arr, /, *, name="Input"):
    """Check if an input can be cast as a NumPy ndarray.

    arr : array_like
        Array to check.

    """
    cast_to_ndarray(arr, name=name)


def check_is_real(arr, /, *, as_warning=False, name="Array"):
    """Raise TypeError if array is not float or integer type.

    Notes
    -----
    Arrays with NaN values are considered real and will not raise
    an error.

    Parameters
    ----------
    arr : array_like
        Array to check.

    as_warning : bool, False
        Issue a UserWarning instead of raising a TypeError.

    """
    check_is_string(name, name="Name")
    arr = cast_to_ndarray(arr, name=name)
    # Do not use np.isreal as it will fail in some cases (e.g. scalars).
    # Check dtype directly instead
    try:
        check_is_subdtype(arr, (np.floating, np.integer), name=name)
    except TypeError as e:
        if as_warning:
            warn(f"{name} does not have real numbers.", UserWarning)
        else:
            raise TypeError(f"{name} must have real numbers.") from e


def check_is_sorted(arr, /, *, name="Array"):
    """Raise ValueError if array is not sorted in ascending order.

    Parameters
    ----------
    arr : array_like
        Array to check.

    """
    check_is_string(name, name="Name")
    arr = cast_to_ndarray(arr, name=name)
    if not np.array_equal(np.sort(arr), arr):
        raise ValueError(f"{name} must be sorted.")


def check_is_finite(arr, /, *, as_warning=False, name="Array"):
    """Raise ValueError if array has any NaN or Inf values.

    Parameters
    ----------
    arr : array_like
        Array to check.

    """
    check_is_string(name, name="Name")
    arr = cast_to_ndarray(arr, name=name)
    if not np.all(np.isfinite(arr)):
        if as_warning:
            warn(f"{name} has non-finite values.", UserWarning)
        else:
            raise ValueError(f"{name} must have finite values.")


def check_is_integerlike(arr, /, *, strict=False, name="Array"):
    """Raise ValueError if any element value differs from its floor.

    The array can have a float or integer data type.

    Parameters
    ----------
    arr : array_like
        Array to check.

    strict : bool, False
        If ``True``, the array's data must be a subtupe of ``np.integer``
        (i.e. float types are not allowed).

    """
    check_is_string(name, name="Name")
    arr = cast_to_ndarray(arr, name=name)
    if strict:
        check_is_subdtype(arr, np.integer, name=name)
    elif not np.array_equal(arr, np.floor(arr)):
        raise ValueError(f"{name} must have integer-like values.")


def check_is_greater_than(arr, /, value, *, strict=True, name="Array"):
    """Check array elements are all greater than some value.

    Raise ValueError if the check fails.

    Parameters
    ----------
    arr : array_like
        Array to check.

    value : Number
        Value which the array's elements must be greater than.

    strict : bool, True
        If ``True``, the array's value must be strictly greater than
        ``value``. Otherwise, values must be greater than or equal to
        ``Value``.

    """
    check_is_string(name, name="Name")

    arr = cast_to_ndarray(arr, name=name)
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

    value : Number
        Value which the array's elements must be less than.

    strict : bool, True
        If ``True``, the array's value must be strictly less than
        ``value``. Otherwise, values must be less than or equal to
        ``Value``.

    Raises
    ------
    ValueError
        If the check fails.

    """
    check_is_string(name, name="Name")
    arr = cast_to_ndarray(arr, name=name)
    if strict and not np.all(arr < value):
        raise ValueError(f"{name} values must all be less than {value}.")
    elif not np.all(arr <= value):
        raise ValueError(f"{name} values must all be less than or equal to {value}.")


def check_is_in_range(arr, /, rng, *, strict_lower=False, strict_upper=False, name="Array"):
    """Check that the array's values are all within a specific range.

    Parameters
    ----------
    arr : array_like
        Array to check.

    rng : array_like, optional
        Array-like with two elements ``[min, max]`` specifying the minimum
        and maximum data values allowed, respectively. By default, the
        range endpoints are inclusive, i.e. values must be >= min
        and <= max. Use ``strict_lower`` and/or ``strict_upper``
        to further restrict the allowable range.

    strict_lower : bool, False
        Enforce a strict lower bound for the range, i.e. array values
        must be strictly greater than the minimum.

    strict_upper : bool, False
        Enforce a strict upper bound for the range, i.e. array values
        must be strictly less than the maximum.

    Raises
    ------
    ValueError if any array value is outside some range.

    """
    # Avoid circular import
    from pyvista.core.input_validation.validate import validate_data_range

    check_is_string(name, name="Name")
    arr = cast_to_ndarray(arr, name=name)
    rng = validate_data_range(rng)

    check_is_greater_than(arr, rng[0], strict=strict_lower, name=name)
    check_is_less_than(arr, rng[1], strict=strict_upper, name=name)


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
        the array must be 1-dimensional with that length. Use a value of
         -1 for an dimension where its size is allowed to vary. Use
         ``()`` to allow scalar values (i.e. 0-dimensional).

    Raises
    ------
    ValueError
        If array does not have any of the specified shape(s).

    """
    from pyvista.core.input_validation.validate import validate_shape_value

    check_is_string(name, name="Name")
    arr = cast_to_ndarray(arr)
    is_error = True
    # NumPy allows shape as an input parameter to be array-like, but
    # here we enforce that shape must be int or tuple because a list is
    # explicitly used as a container for nesting multiple shapes.
    if not isinstance(shape, list):
        shape = [shape]

    for allowable_shape in shape:
        allowable_shape = validate_shape_value(allowable_shape)
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


def check_is_ndarray(arr, /, *, allow_subclass=True, name="Input"):
    """Check that object is a NumPy ndarray."""
    check_is_string(name, name="Name")
    check_is_instance(arr, np.ndarray, allow_subclass=allow_subclass, name=name)


def check_is_number(num, /, *, allow_subclass=True, name='Number'):
    """Check that object is a number."""
    check_is_string(name, name="Name")
    check_is_instance(num, Number, allow_subclass=allow_subclass, name=name)


def check_is_string(obj: str, /, *, allow_subclass=True, name: str = 'Object'):
    """Check that object is a string."""
    check_is_instance(name, str, allow_subclass=True, name="Name")
    check_is_instance(obj, str, allow_subclass=allow_subclass, name=name)


def check_is_sequence(obj: Sequence, /, *, allow_subclass=True, name: str = 'Object'):
    """Check that object is a sequence."""
    check_is_string(name, name="Name")
    check_is_instance(obj, Sequence, allow_subclass=allow_subclass, name=name)


def check_is_iterable(obj: Iterable, /, *, allow_subclass=True, name: str = 'Object'):
    """Check that object is iterable."""
    check_is_string(name, name="Name")
    check_is_instance(obj, Iterable, allow_subclass=allow_subclass, name=name)


def check_is_instance(
    obj, /, classinfo: Union[type, Tuple[type, ...]], *, allow_subclass=True, name: str = 'Object'
):
    """Check that an object is an instance of the given types."""
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
    """Check that object is one of the given types."""
    return check_is_instance(obj, classinfo, allow_subclass=False, name=name)


def check_is_string_sequence():
    """Check that a sequence's elements are all strings."""
    pass


def check_string_is_in_list():
    """Check that string is in a list of strings."""
    pass
