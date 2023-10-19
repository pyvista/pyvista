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
from functools import wraps
from numbers import Number
from typing import Any, Tuple, Union, get_args, get_origin
from warnings import warn

import numpy as np

from pyvista.core._vtk_core import vtkMatrix3x3, vtkMatrix4x4, vtkTransform
from pyvista.core.utilities.arrays import array_from_vtkmatrix

ShapeLike = Union[int, Tuple[int, ...], Tuple[None]]


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


def cast_to_tuple_array(arr, /, *, name="Input"):
    """Cast an array to nested tuples.

    arr : array_like
        Array to check.

    """
    check_is_string(name, name="Name")
    arr = cast_to_ndarray(arr, name=name).tolist()

    def _to_tuple(s):
        return tuple(_to_tuple(i) for i in s) if isinstance(s, list) else s

    return _to_tuple(arr)


def cast_to_list_array(arr, /, *, name="Input"):
    """Cast an array to a nested list.

    arr : array_like
        Array to check.

    """
    check_is_string(name, name="Name")
    return cast_to_ndarray(arr, name=name).tolist()


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


def check_is_dtypelike(dtypelike, /, *, name="Data type"):
    """Check that an input is dtype-like.

    Parameters
    ----------
    dtypelike : dtype-like
        DType-like value to check.

    """
    validate_dtype(dtypelike)


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


def check_is_arraylike(arr, /, *, name="Input"):
    """Check if an input can be cast as a NumPy ndarray.

    arr : array_like
        Array to check.

    """
    cast_to_ndarray(arr, name=name)


def cast_to_ndarray(arr, /, *, as_any=True, dtype=None, copy=False, name="Input") -> np.ndarray:
    """Cast array to a NumPy ndarray.

    Parameters
    ----------
    arr : array_like
        Array to check.

    Raises
    ------
    ValueError
        If input cannot be cast as a NumPy ndarray.

    Returns
    -------
    np.ndarray

    """
    check_is_string(name, name="Name")
    try:
        if as_any:
            out = np.asanyarray(arr, dtype=dtype)
            if copy:
                out = out.copy()
        else:
            out = np.array(arr, dtype=dtype, copy=copy)
        if out.dtype.name == 'object':
            # NumPy will normally raise ValueError automatically for
            # object arrays, but on some systems it will not, so raise
            # error manually
            raise ValueError
    except (ValueError, np.VisibleDeprecationWarning) as e:
        raise ValueError(f"{name} cannot be cast as {np.ndarray}.") from e
    return out


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
    check_is_string(name, name="Name")
    check_is_instance(arr, np.ndarray, allow_subclass=allow_subclass, name=name)


def check_is_number(num, /, *, allow_subclass=True, name='Number'):
    check_is_string(name, name="Name")
    check_is_instance(num, Number, allow_subclass=allow_subclass, name=name)


@wraps(validate_numeric_array)
def validate_number(num, **kwargs):
    kwargs.setdefault('name', 'Number')
    kwargs.setdefault('to_list', True)
    _set_default_kwarg_mandatory(kwargs, 'shape', (), name="Number")
    _set_default_kwarg_mandatory(kwargs, 'must_be_real', True)
    return validate_numeric_array(num, **kwargs)


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


def validate_data_range(rng, **kwargs):
    """Validate a data range."""
    name = 'Data Range'
    kwargs.setdefault('name', name)
    _set_default_kwarg_mandatory(kwargs, 'shape', 2, name=name)
    _set_default_kwarg_mandatory(kwargs, 'must_be_sorted', True, name=name)
    if 'to_list' not in kwargs:
        kwargs.setdefault('to_tuple', True)
    return validate_numeric_array(rng, **kwargs)


def validate_arrayNx3(arr, **kwargs):
    """Validate an array is numeric and has Nx3 shape."""
    name = 'Array'
    kwargs.setdefault('name', name)
    _set_default_kwarg_mandatory(kwargs, 'shape', [(-1, 3)], name=name)
    return validate_numeric_array(arr, **kwargs)


def coerce_array_to_arrayNx3(arr, **kwargs):
    """Validate an array is numeric and ensure out has Nx3 shape.
    Input array can be 1D array-like with 3 elements or array-like with
    shape Nx3. The output is reshaped to ensure it is 2D with Nx3 shape.
    """
    name = 'Array'
    kwargs.setdefault('name', name)
    _set_default_kwarg_mandatory(kwargs, 'shape', [3, (-1, 3)], name=name)
    return validate_numeric_array(arr, **kwargs).reshape((-1, 3))


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


def coerce_number_or_array3_as_array3():
    """Check that a sequence's elements are all strings."""
    pass


def validate_numeric_array_shape3():
    pass


def check_string_is_in_list():
    pass
