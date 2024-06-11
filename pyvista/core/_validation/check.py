"""Functions that check the type and/or value of inputs.

.. versionadded:: 0.43.0

A ``check`` function typically:

* Performs a simple validation on a single input variable.
* Raises an error if the check fails due to invalid input.
* Does not modify input or return anything.

"""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Number
import reprlib
from typing import TYPE_CHECKING
from typing import Sequence
from typing import Union
from typing import get_args
from typing import get_origin

import numpy as np
import numpy.typing as npt

from pyvista.core._validation._cast_array import _cast_to_numpy

if TYPE_CHECKING:  # pragma: no cover
    from pyvista.core._typing_core import NumberType
    from pyvista.core._typing_core._aliases import _ArrayLikeOrScalar


def check_subdtype(
    input_obj: Union[npt.DTypeLike, _ArrayLikeOrScalar[NumberType]],
    /,
    base_dtype: Union[npt.DTypeLike, tuple[npt.DTypeLike, ...], list[npt.DTypeLike]],
    *,
    name: str = 'Input',
):
    """Check if an input's data-type is a subtype of another data-type(s).

    Parameters
    ----------
    input_obj : float | ArrayLike[float] | numpy.typing.DTypeLike
        ``dtype`` object (or object coercible to one) or an array-like object.
        If array-like, the dtype of the array is used.

    base_dtype : numpy.typing.DTypeLike | Sequence[numpy.typing.DTypeLike]
        ``dtype``-like object or a sequence of ``dtype``-like objects. The ``input_obj``
        must be a subtype of this value. If a sequence, ``input_obj`` must be a
        subtype of at least one of the specified dtypes.

    name : str, default: "Input"
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    TypeError
        If ``input_obj`` is not a subtype of ``base_dtype``.

    See Also
    --------
    check_real
    check_number

    Examples
    --------
    Check if ``float`` is a subtype of ``np.floating``.

    >>> import numpy as np
    >>> from pyvista import _validation
    >>> _validation.check_subdtype(float, np.floating)

    Check from multiple allowable dtypes.

    >>> _validation.check_subdtype(int, [np.integer, np.floating])

    Check an array's dtype.

    >>> array = np.array([1, 2, 3], dtype='uint8')
    >>> _validation.check_subdtype(array, np.integer)

    """
    input_dtype: npt.DTypeLike
    try:
        input_dtype = np.dtype(input_obj)  # type: ignore[arg-type]
    except TypeError:
        input_dtype = np.asanyarray(input_obj).dtype

    if not isinstance(base_dtype, (tuple, list)):
        base_dtype = [base_dtype]

    if not any(np.issubdtype(input_dtype, base) for base in base_dtype):
        # Not a subdtype, so raise error
        msg = f"{name} has incorrect dtype of '{input_dtype.name}'. "
        if len(base_dtype) == 1:
            msg += f"The dtype must be a subtype of {base_dtype[0]}."
        else:
            msg += f"The dtype must be a subtype of at least one of \n{base_dtype}."
        raise TypeError(msg)


def check_real(array: _ArrayLikeOrScalar[NumberType], /, *, name: str = "Array"):
    """Check if an array has real numbers, i.e. float or integer type.

    Notes
    -----
    -   Boolean data types are not considered real and will raise an error.
    -   Arrays with ``infinity`` or ``NaN`` values are considered real and
        will not raise an error. Use :func:`check_finite` to check for
        finite values.

    Parameters
    ----------
    array : float | ArrayLike[float]
        Number or array to check.

    name : str, default: "Array"
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    TypeError
        If the array does not have real numbers.

    See Also
    --------
    check_integer
        Similar function for integer arrays.
    check_number
        Similar function for scalar values.
    check_finite
        Check for finite values.

    Examples
    --------
    Check if an array has real numbers.

    >>> from pyvista import _validation
    >>> _validation.check_real([1, 2, 3])

    """
    array = array if isinstance(array, np.ndarray) else _cast_to_numpy(array)

    # Return early for common cases
    if array.dtype.type in [np.int32, np.int64, np.float32, np.float64]:
        return

    # Do not use np.isreal as it will fail in some cases (e.g. scalars).
    # Check dtype directly instead
    try:
        check_subdtype(array, (np.floating, np.integer), name=name)
    except TypeError as e:
        raise TypeError(f"{name} must have real numbers.") from e


def check_sorted(
    array: _ArrayLikeOrScalar[NumberType],
    /,
    *,
    ascending: bool = True,
    strict: bool = False,
    axis: int = -1,
    name: str = "Array",
):
    """Check if an array's values are sorted.

    Parameters
    ----------
    array : float | ArrayLike[float]
        Number or array to check.

    ascending : bool, default: True
        If ``True``, check if the array's elements are in ascending order.
        If ``False``, check if the array's elements are in descending order.

    strict : bool, default: False
        If ``True``, the array's elements must be strictly increasing (if
        ``ascending=True``) or strictly decreasing (if ``ascending=False``).
        Effectively, this means the array must be sorted *and* its values
        must be unique.

    axis : int | None, default: -1
        Axis along which to check sorting. If ``None``, the array is flattened
        before checking. The default is ``-1``, which checks sorting along the
        last axis.

    name : str, default: "Array"
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    ValueError
        If the array is not sorted in ascending order.

    See Also
    --------
    check_range

    Examples
    --------
    Check if an array's values are sorted,

    >>> from pyvista import _validation
    >>> _validation.check_sorted([1, 2, 3])

    """
    array = array if isinstance(array, np.ndarray) else _cast_to_numpy(array)

    ndim = array.ndim
    if ndim == 0:
        # Scalars are always sorted
        return

    # Validate axis
    if axis not in [-1, None]:
        check_number(axis, name="Axis")
        check_integer(axis, name="Axis")
        axis = int(axis)
        try:
            check_range(axis, rng=[-ndim, ndim - 1], name="Axis")
        except ValueError:
            raise ValueError(f"Axis {axis} is out of bounds for ndim {ndim}.")

    if axis is None and ndim >= 1:
        # Emulate np.sort(), which flattens array when axis is None
        array = array.ravel(order='A')
        ndim = 1
        axis = 0

    # Create slicers to get a view along an axis
    # Create two slicers to compare consecutive elements with each other
    first_slice = [slice(None)] * ndim
    first_slice[axis] = slice(None, -1)
    first_item = array[tuple(first_slice)]

    second_slice = [slice(None)] * ndim
    second_slice[axis] = slice(1, None)
    second_item = array[tuple(second_slice)]

    if ascending and not strict:
        is_sorted = np.all(first_item <= second_item)
    elif ascending and strict:
        is_sorted = np.all(first_item < second_item)
    elif not ascending and not strict:
        is_sorted = np.all(first_item >= second_item)
    else:  # not ascending and strict
        is_sorted = np.all(first_item > second_item)

    if not is_sorted:
        # Show the array's elements in error msg if array is small
        msg_body = f"with {array.size} elements"
        order = "ascending" if ascending else "descending"
        strict_ = "strict " if strict else ""
        raise ValueError(
            f"{name} {msg_body} must be sorted in {strict_}{order} order. "
            f"Got:\n    {reprlib.repr(array)}",
        )


def check_finite(arr, /, *, name="Array"):
    """Check if an array has finite values, i.e. no NaN or Inf values.

    Parameters
    ----------
    arr : array_like
        Array to check.

    name : str, default: "Array"
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    ValueError
        If the array has any ``Inf`` or ``NaN`` values.

    See Also
    --------
    check_real

    Examples
    --------
    Check if an array's values are finite.

    >>> from pyvista import _validation
    >>> _validation.check_finite([1, 2, 3])

    """
    arr = arr if isinstance(arr, np.ndarray) else _cast_to_numpy(arr)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must have finite values.")


def check_integer(arr, /, *, strict=False, name="Array"):
    """Check if an array has integer or integer-like float values.

    Parameters
    ----------
    arr : array_like
        Array to check.

    strict : bool, default: False
        If ``True``, the array's data must be a subtype of ``np.integer``
        (i.e. float types are not allowed).

    name : str, default: "Array"
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    ValueError
        If any element's value differs from its floor.

    TypeError
        If ``strict=True`` and the array's dtype is not integral.

    See Also
    --------
    check_nonnegative

    Examples
    --------
    Check if an array has integer-like values.

    >>> from pyvista import _validation
    >>> _validation.check_integer([1.0, 2.0])

    """
    arr = arr if isinstance(arr, np.ndarray) else _cast_to_numpy(arr)
    if strict:
        check_subdtype(arr, np.integer)
    elif not np.array_equal(arr, np.floor(arr)):
        raise ValueError(f"{name} must have integer-like values.")


def check_nonnegative(arr, /, *, name="Array"):
    """Check if an array's elements are all nonnegative.

    Parameters
    ----------
    arr : array_like
        Array to check.

    name : str, default: "Array"
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    ValueError
        If the array has any negative values.

    See Also
    --------
    check_greater_than
    check_less_than

    Examples
    --------
    Check if an array's values are non-negative.

    >>> from pyvista import _validation
    >>> _validation.check_nonnegative([1, 2, 3])

    """
    check_greater_than(arr, 0, strict=False, name=name)


def check_greater_than(arr, /, value, *, strict=True, name="Array"):
    """Check if an array's elements are all greater than some value.

    Parameters
    ----------
    arr : array_like
        Array to check.

    value : Number
        Value which the array's elements must be greater than.

    strict : bool, default: True
        If ``True``, the array's value must be strictly greater than ``value``.
        Otherwise, values must be greater than or equal to ``value``.

    name : str, default: "Array"
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    ValueError
        If not all array elements are greater than (or equal to if
        ``strict=True``) the specified value.

    See Also
    --------
    check_less_than
    check_in_range
    check_nonnegative

    Examples
    --------
    Check if an array's values are greater than 0.

    >>> from pyvista import _validation
    >>> _validation.check_greater_than([1, 2, 3], value=0)

    """
    arr = arr if isinstance(arr, np.ndarray) else _cast_to_numpy(arr)
    value = _cast_to_numpy(value)
    check_shape(value, ())
    check_real(value)
    check_finite(value)
    if strict and not np.all(arr > value):
        raise ValueError(f"{name} values must all be greater than {value}.")
    elif not np.all(arr >= value):
        raise ValueError(f"{name} values must all be greater than or equal to {value}.")


def check_less_than(arr, /, value, *, strict=True, name="Array"):
    """Check if an array's elements are all less than some value.

    Parameters
    ----------
    arr : array_like
        Array to check.

    value : Number
        Value which the array's elements must be less than.

    strict : bool, default: True
        If ``True``, the array's value must be strictly less than
        ``value``. Otherwise, values must be less than or equal to
        ``value``.

    name : str, default: "Array"
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    ValueError
        If not all array elements are less than (or equal to if
        ``strict=True``) the specified value.

    See Also
    --------
    check_greater_than
    check_in_range
    check_nonnegative

    Examples
    --------
    Check if an array's values are less than 0.

    >>> from pyvista import _validation
    >>> _validation.check_less_than([-1, -2, -3], value=0)

    """
    arr = arr if isinstance(arr, np.ndarray) else _cast_to_numpy(arr)
    if strict and not np.all(arr < value):
        raise ValueError(f"{name} values must all be less than {value}.")
    elif not np.all(arr <= value):
        raise ValueError(f"{name} values must all be less than or equal to {value}.")


def check_range(arr, /, rng, *, strict_lower=False, strict_upper=False, name="Array"):
    """Check if an array's values are all within a specific range.

    Parameters
    ----------
    arr : array_like
        Array to check.

    rng : array_like[float, float], optional
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

    name : str, default: "Array"
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    ValueError
        If any array value is outside the specified range.

    See Also
    --------
    check_less_than
    check_greater_than

    Examples
    --------
    Check if `an array's values are in the range ``[0, 1]``.

    >>> from pyvista import _validation
    >>> _validation.check_range([0, 0.5, 1], rng=[0, 1])

    """
    rng = _cast_to_numpy(rng)
    check_shape(rng, 2, name="Range")
    check_real(rng, name="Range")
    check_sorted(rng, name="Range")

    arr = arr if isinstance(arr, np.ndarray) else _cast_to_numpy(arr)
    check_greater_than(arr, rng[0], strict=strict_lower, name=name)
    check_less_than(arr, rng[1], strict=strict_upper, name=name)


def check_shape(
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

    shape : int, tuple[int, ...] | list[int, tuple[int, ...]], optional
        A single shape or a list of any allowable shapes. If an integer,
        ``i``, the shape is interpreted as ``(i,)``. Use a value of
        -1 for any dimension where its size is allowed to vary, e.g.
        ``(-1,3)`` if any Nx3 array is allowed. Use ``()`` for the
        shape of scalar values (i.e. 0-dimensional). If a list, the
        array must have at least one of the specified shapes.

    name : str, default: "Array"
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    ValueError
        If the array does not have any of the specified shape(s).

    See Also
    --------
    check_length

    Examples
    --------
    Check if an array is one-dimensional.

    >>> import numpy as np
    >>> from pyvista import _validation
    >>> _validation.check_shape([1, 2, 3], shape=(-1))

    Check if an array is one-dimensional or a scalar.

    >>> _validation.check_shape(1, shape=[(), (-1)])

    Check if an array is 3x3 or 4x4.

    >>> _validation.check_shape(np.eye(3), shape=[(3, 3), (4, 4)])

    """

    def _shape_is_allowed(a, b):
        # a: array's actual shape
        # b: allowed shape (may have -1)
        return bool(len(a) == len(b) and all(map(lambda x, y: True if x == y else y == -1, a, b)))

    arr = arr if isinstance(arr, np.ndarray) else _cast_to_numpy(arr)

    if not isinstance(shape, list):
        shape = [shape]

    array_shape = arr.shape
    for shp in shape:
        shp = _validate_shape_value(shp)
        if _shape_is_allowed(array_shape, shp):
            return

    msg = f"{name} has shape {arr.shape} which is not allowed. "
    if len(shape) == 1:
        msg += f"Shape must be {shape[0]}."
    else:
        msg += f"Shape must be one of {shape}."
    raise ValueError(msg)


def check_number(num, /, *, name='Object'):
    """Check if an object is an instance of ``Number``.

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

    See Also
    --------
    check_scalar

    Examples
    --------
    Check if a complex number is an instance of ``Number``.

    >>> from pyvista import _validation
    >>> _validation.check_number(1 + 2j)

    """
    check_instance(num, Number, allow_subclass=True, name=name)


def check_string(obj, /, *, allow_subclass=True, name='Object'):
    """Check if an object is an instance of ``str``.

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

    See Also
    --------
    check_contains
    check_iterable_items
    check_sequence
    check_instance

    Examples
    --------
    Check if an object is a string.

    >>> from pyvista import _validation
    >>> _validation.check_string("eggs")

    """
    check_instance(obj, str, allow_subclass=allow_subclass, name=name)


def check_sequence(obj, /, *, name='Object'):
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

    See Also
    --------
    check_iterable
    check_instance

    Examples
    --------
    Check if an object is a sequence.

    >>> import numpy as np
    >>> from pyvista import _validation
    >>> _validation.check_sequence([1, 2, 3])
    >>> _validation.check_sequence("A")

    """
    check_instance(obj, Sequence, allow_subclass=True, name=name)


def check_iterable(obj, /, *, name='Object'):
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

    See Also
    --------
    check_sequence
    check_instance
    check_iterable_items

    Examples
    --------
    Check if an object is iterable.

    >>> import numpy as np
    >>> from pyvista import _validation
    >>> _validation.check_iterable([1, 2, 3])
    >>> _validation.check_iterable(np.array((4, 5, 6)))

    """
    check_instance(obj, Iterable, allow_subclass=True, name=name)


def check_instance(obj, /, classinfo, *, allow_subclass=True, name='Object'):
    """Check if an object is an instance of the given type or types.

    Parameters
    ----------
    obj : Any
        Object to check.

    classinfo : type | tuple[type, ...]
        ``type`` or tuple of types. Object must be an instance of one of
        the types.

    allow_subclass : bool, default: True
        If ``True``, the object's type must be specified by ``classinfo``
         or any of its subclasses. Otherwise, subclasses are not allowed.

    name : str, default: "Object"
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    TypeError
        If object is not an instance of any of the given types.

    See Also
    --------
    check_type
    check_number
    check_string
    check_iterable
    check_sequence

    Examples
    --------
    Check if an object is an instance of ``complex``.

    >>> from pyvista import _validation
    >>> _validation.check_instance(1 + 2j, complex)

    Check if an object is an instance of one of several types.

    >>> _validation.check_instance("eggs", (int, str))

    """
    if not isinstance(name, str):
        raise TypeError(f"Name must be a string, got {type(name)} instead.")

    # Get class info from generics
    if get_origin(classinfo) is Union:
        classinfo = get_args(classinfo)

    # Count num classes
    num_classes = len(classinfo) if isinstance(classinfo, tuple) else 1

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


def check_type(obj, /, classinfo, *, name='Object'):
    """Check if an object is one of the given type or types.

    Notes
    -----
    The use of :func:`check_instance` is generally preferred as it
    allows subclasses. Use :func:`check_type` only for cases where
    exact types are necessary.

    Parameters
    ----------
    obj : Any
        Object to check.

    classinfo : type | tuple[type, ...]
        ``type`` or tuple of types. Object must be one of the types.

    name : str, default: "Object"
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    TypeError
        If object is not any of the given types.

    See Also
    --------
    check_instance

    Examples
    --------
    Check if an object is type ``dict`` or ``set``.

    >>> from pyvista import _validation
    >>> _validation.check_type({'spam': "eggs"}, (dict, set))

    """
    check_instance(obj, classinfo, allow_subclass=False, name=name)


def check_iterable_items(
    iterable_obj,
    /,
    item_type,
    *,
    allow_subclass=True,
    name='Iterable',
):
    """Check if an iterable's items all have a specified type.

    Parameters
    ----------
    iterable_obj : Iterable
        Iterable to check.

    item_type : type | tuple[type, ...]
        Class type(s) to check for. Each element of the sequence must
        have the type or one of the types specified.

    allow_subclass : bool, default: True
        If ``True``, the type of the iterable items must be any of the
        given types or a subclass thereof. Otherwise, subclasses are not
        allowed.

    name : str, default: "Iterable"
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    TypeError
        If any of the items in the iterable have an incorrect type.

    See Also
    --------
    check_instance
    check_iterable
    check_iterable_items

    Examples
    --------
    Check if a ``tuple`` only has ``int`` or ``float`` elements.

    >>> from pyvista import _validation
    >>> _validation.check_iterable_items((1, 2, 3.0), (int, float))

    Check if a ``list`` only has ``list`` elements.

    >>> from pyvista import _validation
    >>> _validation.check_iterable_items([[1], [2], [3]], list)

    """
    check_iterable(iterable_obj, name=name)
    any(
        check_instance(
            item,
            item_type,
            allow_subclass=allow_subclass,
            name=f"All items of {name}",
        )
        for item in iterable_obj
    )


def check_contains(*, item, container, name='Input'):
    """Check if an item is in a container.

    Parameters
    ----------
    item : Any
        Item to check.

    container : Any
        Container the item is expected to be in.

    name : str, default: "Input"
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    ValueError
        If the string is not in the iterable.

    See Also
    --------
    check_iterable
    check_iterable_items

    Examples
    --------
    Check if ``"A"`` is in a list of strings.

    >>> from pyvista import _validation
    >>> _validation.check_contains(item="A", container=["A", "B", "C"])

    """
    if item not in container:
        qualifier = 'one of' if isinstance(container, (list, tuple)) else 'in'
        msg = f"{name} '{item}' is not valid. {name} must be {qualifier}: \n\t{container}"
        raise ValueError(msg)


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
    """Check if the length of an array meets specific requirements.

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

    exact_length : int | array_like[int, ...]
        Check if the array has the given length. If multiple
        numbers are given, the array's length must match one of the
        numbers.

    min_length : int, optional
        Check if the array has this length or greater.

    max_length : int, optional
        Check if the array has this length or less.

    must_be_1d : bool, default: False
        If ``True``, check if the shape of the array is one-dimensional,
        i.e. that the array's shape is ``(1,)``.

    allow_scalars : bool, default: False
        If ``True``, a scalar input will be reshaped to have a length of
        1. Otherwise, the check will fail since a scalar does not
        have a length.

    name : str, default: "Array"
        Variable name to use in the error messages if any are raised.

    Raises
    ------
    ValueError
        If the array's length is outside the specified range.

    See Also
    --------
    check_shape

    Examples
    --------
    Check if an array has a length of 2 or 3.

    >>> from pyvista import _validation
    >>> _validation.check_length([1, 2], exact_length=[2, 3])

    Check if an array has a minimum length of 3.

    >>> _validation.check_length((1, 2, 3), min_length=3)

    Check if a multidimensional array has a maximum length of 2.

    >>> _validation.check_length([[1, 2, 3], [4, 5, 6]], max_length=2)

    """
    if allow_scalars:
        # Reshape to 1D
        if isinstance(arr, np.ndarray) and arr.ndim == 0:
            arr = [arr.tolist()]
        elif isinstance(arr, Number):
            arr = [arr]
    check_instance(arr, (Sequence, np.ndarray), name=name)

    if must_be_1d:
        check_shape(arr, shape=(-1))

    if exact_length is not None:
        exact_length = np.array(exact_length)
        check_integer(exact_length, name="'exact_length'")
        if len(arr) not in exact_length:
            raise ValueError(
                f"{name} must have a length equal to any of: {exact_length}. "
                f"Got length {len(arr)} instead.",
            )

    # Validate min/max length
    if min_length is not None:
        min_length = _cast_to_numpy(min_length)
        check_number(min_length.tolist(), name="Min length")
        check_real(min_length, name="Min length")
    if max_length is not None:
        max_length = _cast_to_numpy(max_length)
        check_number(max_length.tolist(), name="Max length")
        check_real(max_length, name="Max length")
    if min_length is not None and max_length is not None:
        check_sorted((min_length, max_length), name="Range")

    if min_length is not None:
        if len(arr) < min_length:
            raise ValueError(
                f"{name} must have a minimum length of {min_length}. "
                f"Got length {len(arr)} instead.",
            )
    if max_length is not None:
        if len(arr) > max_length:
            raise ValueError(
                f"{name} must have a maximum length of {max_length}. "
                f"Got length {len(arr)} instead.",
            )


def _validate_shape_value(shape: int | tuple[int, ...] | tuple[None]):
    """Validate shape-like input and return its tuple representation."""
    if shape is None:
        # `None` is used to mean `any shape is allowed` by the array
        #  validation methods, so raise an error here.
        #  Also, setting `None` as a shape is deprecated by NumPy.
        raise TypeError("`None` is not a valid shape. Use `()` instead.")

    # Return early for common inputs
    if shape in [(), (-1,), (1,), (3,), (2,), (1, 3), (-1, 3)]:
        return shape

    def _is_valid_dim(d):
        return isinstance(d, int) and d >= -1

    if _is_valid_dim(shape):
        return (shape,)
    if isinstance(shape, tuple) and all(map(_is_valid_dim, shape)):
        return shape

    # Input is not valid at this point. Use checks to raise an
    # appropriate error
    check_instance(shape, (int, tuple), name='Shape')
    if isinstance(shape, int):
        shape = (shape,)
    else:
        check_iterable_items(shape, int, name='Shape')
    check_greater_than(shape, -1, name="Shape", strict=False)
    raise RuntimeError("This line should not be reachable.")  # pragma: no cover
