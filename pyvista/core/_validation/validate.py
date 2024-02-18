"""Functions that validate input and return a standard representation.

.. versionadded:: 0.44.0

A ``validate`` function typically:

* Uses :py:mod:`~pyvista.core._validation.check` functions to
  check the type and/or value of input arguments.
* Applies (optional) constraints, e.g. input or output must have a
  specific length, shape, type, data-type, etc.
* Accepts many different input types or values and standardizes the
  output as a single representation with known properties.

"""

from __future__ import annotations

from collections import namedtuple
import inspect
from itertools import product
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union, overload

import numpy as np
from typing_extensions import TypedDict, Unpack

from pyvista.core import _vtk_core as _vtk
from pyvista.core._typing_core import Matrix, NumpyArray, TransformLike, Vector
from pyvista.core._typing_core._array_like import (
    __NumberType,
    _ArrayLike,
    _ArrayLikeOrScalar,
    _FiniteNestedList,
    _FiniteNestedTuple,
    _NumberType,
    _NumberUnion,
)
from pyvista.core._validation._array_wrapper import _ArrayLikeWrapper, _BuiltinWrapper
from pyvista.core._validation.check import (
    check_contains,
    check_finite,
    check_integer,
    check_length,
    check_nonnegative,
    check_range,
    check_real,
    check_shape,
    check_sorted,
    check_subdtype,
)

_ValidationFlags = namedtuple(
    '_ValidationFlags', ['same_shape', 'same_dtype', 'same_type', 'same_object']
)

_FloatType = TypeVar('_FloatType', bound=float)
_ShapeLike = Union[int, Tuple[int, ...], Tuple[()]]

_NumpyReturnType = Union[
    Literal['numpy'],
    Type[np.ndarray],  # type: ignore[type-arg]
]
_ListReturnType = Union[
    Literal['list'],
    Type[list],  # type: ignore[type-arg]
]
_TupleReturnType = Union[
    Literal['tuple'],
    Type[tuple],  # type: ignore[type-arg]
]
_ArrayReturnType = Union[_NumpyReturnType, _ListReturnType, _TupleReturnType]


class _TypedKwargs(TypedDict, total=False):
    must_have_shape: Optional[Union[_ShapeLike, List[_ShapeLike]]]
    must_have_dtype: Optional[_NumberUnion]
    must_have_length: Optional[Union[int, Vector[int]]]
    must_have_min_length: Optional[int]
    must_have_max_length: Optional[int]
    must_be_nonnegative: bool
    must_be_finite: bool
    must_be_real: bool
    must_be_integer: bool
    must_be_sorted: Union[bool, Dict[str, Union[bool, int]]]
    must_be_in_range: Optional[Vector[float]]
    strict_lower_bound: bool
    strict_upper_bound: bool
    reshape_to: Optional[_ShapeLike]
    broadcast_to: Optional[_ShapeLike]
    as_any: bool
    copy: bool
    get_flags: bool
    name: str


# """SCALAR OVERLOADS"""
# T -> T
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: _NumberType,
    /,
    *,
    dtype_out: None = None,
    return_type: Optional[Union[_TupleReturnType, _ListReturnType]] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> _NumberType: ...


# T1 -> T2
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: _NumberType,
    /,
    *,
    dtype_out: Type[__NumberType] = ...,
    return_type: Optional[Union[_TupleReturnType, _ListReturnType]] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> __NumberType: ...


# T -> NDArray[T]
@overload
def validate_array(  # numpydoc ignore=GL08
    array: _NumberType,
    /,
    *,
    dtype_out: None = None,
    return_type: Optional[_NumpyReturnType] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> NumpyArray[_NumberType]: ...


# """LIST OVERLOADS"""
# List[List[T]] -> List[List[T]]
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: List[List[_NumberType]],
    /,
    *,
    dtype_out: None = None,
    return_type: Optional[_ListReturnType] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> List[List[_NumberType]]: ...


# List[List[T1]] -> List[List[T2]]
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: List[List[_NumberType]],
    /,
    *,
    dtype_out: Type[__NumberType] = ...,
    return_type: Optional[_ListReturnType] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> List[List[__NumberType]]: ...


# List[List[T]] -> Tuple[Tuple[T]]
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: List[List[_NumberType]],
    /,
    *,
    dtype_out: None = None,
    return_type: _TupleReturnType = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> Tuple[Tuple[_NumberType]]: ...


# List[List[T1]] -> Tuple[Tuple[T2]]
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: List[List[_NumberType]],
    /,
    *,
    dtype_out: Type[__NumberType] = ...,
    return_type: _TupleReturnType = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> Tuple[Tuple[__NumberType]]: ...


# List[T] -> List[T]
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: List[_NumberType],
    /,
    *,
    dtype_out: None = None,
    return_type: Optional[_ListReturnType] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> List[_NumberType]: ...


# List[T1] -> List[T2]
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: List[_NumberType],
    /,
    *,
    dtype_out: Type[__NumberType] = ...,
    return_type: Optional[_ListReturnType] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> List[__NumberType]: ...


# List[T] -> Tuple[T]
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: List[_NumberType],
    /,
    *,
    dtype_out: None = None,
    return_type: _TupleReturnType = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> Tuple[_NumberType]: ...


# List[T1] -> Tuple[T2]
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: List[_NumberType],
    /,
    *,
    dtype_out: Type[__NumberType] = ...,
    return_type: _TupleReturnType = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> Tuple[__NumberType]: ...


# """TUPLE OVERLOADS"""
# Tuple[Tuple[T]] -> Tuple[Tuple[T]]
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: Tuple[Tuple[_NumberType]],
    /,
    *,
    dtype_out: None = None,
    return_type: Optional[_TupleReturnType] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> Tuple[Tuple[_NumberType]]: ...


# Tuple[Tuple[T]] -> List[List[T]]
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: Tuple[Tuple[_NumberType]],
    /,
    *,
    dtype_out: None = None,
    return_type: _ListReturnType = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> List[List[_NumberType]]: ...


# Tuple[Tuple[T1]] -> Tuple[Tuple[T2]]
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: Tuple[Tuple[_NumberType]],
    /,
    *,
    dtype_out: Type[__NumberType] = ...,
    return_type: Optional[_TupleReturnType] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> Tuple[Tuple[__NumberType]]: ...


# Tuple[T] -> Tuple[T]
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: Tuple[_NumberType, ...],
    /,
    *,
    dtype_out: None = None,
    return_type: Optional[_TupleReturnType] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> Tuple[_NumberType]: ...


# Tuple[T] -> List[T]
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: Tuple[_NumberType],
    /,
    *,
    dtype_out: None = None,
    return_type: _ListReturnType = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> List[_NumberType]: ...


# Tuple[T1] -> Tuple[T2]
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: Tuple[_NumberType, ...],
    /,
    *,
    dtype_out: Type[__NumberType] = ...,
    return_type: Optional[_TupleReturnType] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> Tuple[__NumberType]: ...


# """NUMPY OVERLOADS"""
# NDArray[T] -> NDArray[T]
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: NumpyArray[_NumberType],
    /,
    *,
    dtype_out: None = None,
    return_type: Optional[_NumpyReturnType] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> NumpyArray[_NumberType]: ...


# NDArray[T1] -> NDArray[T2]
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: NumpyArray[_NumberType],
    /,
    *,
    dtype_out: Type[__NumberType] = ...,
    return_type: Optional[_NumpyReturnType] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> NumpyArray[__NumberType]: ...


# NDArray[T] -> NestedList[T]
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: NumpyArray[_NumberType],
    /,
    *,
    dtype_out: None = None,
    return_type: _ListReturnType = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> _FiniteNestedList[_NumberType]: ...


# NDArray[T] -> NestedTuple[T]
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: NumpyArray[_NumberType],
    /,
    *,
    dtype_out: None = None,
    return_type: _TupleReturnType = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> _FiniteNestedTuple[_NumberType]: ...


# """ARRAY-LIKE OVERLOADS"""
# These are general catch-all cases for anything not overloaded explicitly
# ArrayLike[T] -> NDArray[T]
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: _ArrayLike[_NumberType],
    /,
    *,
    dtype_out: None = None,
    return_type: Optional[_NumpyReturnType] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> NumpyArray[_NumberType]: ...


# ArrayLike[T1] -> NDArray[T2]
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: _ArrayLike[_NumberType],
    /,
    *,
    dtype_out: Type[__NumberType] = ...,
    return_type: Optional[_NumpyReturnType] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> NumpyArray[__NumberType]: ...


# ArrayLike[T] -> NestedList[T]
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: _ArrayLike[_NumberType],
    /,
    *,
    dtype_out: None = None,
    return_type: _ListReturnType = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> _FiniteNestedList[_NumberType]: ...


# ArrayLike[T] -> NestedTuple[T]
@overload
def validate_array(  # numpydoc ignore=GL08
    array: _ArrayLike[_NumberType],
    /,
    *,
    dtype_out: None = None,
    return_type: _TupleReturnType = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> _FiniteNestedTuple[_NumberType]: ...


@overload
def validate_array(  # numpydoc ignore=GL08
    array: _ArrayLike[_NumberType],
    /,
    *,
    dtype_out: Type[__NumberType] = ...,
    return_type: _TupleReturnType = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> _FiniteNestedTuple[__NumberType]: ...


def validate_array(
    array: _ArrayLikeOrScalar[_NumberType],
    /,
    *,
    must_have_shape: Optional[Union[_ShapeLike, List[_ShapeLike]]] = None,
    must_have_dtype: Optional[_NumberUnion] = None,
    must_have_length: Optional[Union[int, Vector[int]]] = None,
    must_have_min_length: Optional[int] = None,
    must_have_max_length: Optional[int] = None,
    must_be_nonnegative: bool = False,
    must_be_finite: bool = False,
    must_be_real: bool = True,
    must_be_integer: bool = False,
    must_be_sorted: Union[bool, Dict[str, Union[bool, int]]] = False,
    must_be_in_range: Optional[Vector[float]] = None,
    strict_lower_bound: bool = False,
    strict_upper_bound: bool = False,
    reshape_to: Optional[_ShapeLike] = None,
    broadcast_to: Optional[_ShapeLike] = None,
    dtype_out: Optional[Type[__NumberType]] = None,
    return_type: Optional[_ArrayReturnType] = None,
    as_any: bool = True,
    copy: bool = False,
    get_flags: bool = False,
    name: str = 'Array',
):
    """Check and validate a numeric array meets specific requirements.

    Validate an array to ensure it is numeric, has a specific shape,
    data-type, and/or has values that meet specific requirements such as
    being sorted, having integer values, or is finite. The array can
    optionally be reshaped or broadcast, and the output type of
    the array can be explicitly set to standardize its representation.

    By default, this function is generic and returns an array with the
    same type and dtype as the input array, i.e. ``Array[T] -> Array[T]``
    for the common cases below. In most cases, it will also return the
    array as-is without copying its data wherever possible.

    - Scalars: ``T`` -> ``T``
    - Lists: ``List[T]`` -> ``List[T]``
    - Nested lists: ``List[List[T]]`` -> ``List[List[T]]``
    - Tuples: ``Tuple[T]`` -> ``Tuple[T]``
    - Nested tuples: ``Tuple[Tuple[T]]`` -> ``Tuple[Tuple[T]]``
    - NumPy arrays: ``NDArray[T]`` -> ``NDArray[T]``

    For all other input types, (e.g. ``range`` objects, nested lists with
    depth > 2, NumPy protocol arrays, etc.), the returned array is a NumPy
    array.

    Optionally, use ``return_type`` and/or ``dtype_out`` for non-generic
    behavior to ensure the output array has a consistent type.

    .. warning::

        This function is primarily designed to work with homogeneous
        numeric arrays with a regular shape. Any other array-like
        inputs (e.g. structured arrays, string arrays) are not
        supported.

    .. note::

        This function is very general. For common validation workflows, it is
        recommended to use a wrapper function instead with pre-set default
        parameters, or use one of the specialized functions below.

    See Also
    --------
    validate_number
        Specialized function for single numbers.

    validate_array3
        Specialized function for 3-element arrays.

    validate_arrayN
        Specialized function for one-dimensional arrays.

    validate_arrayN_unsigned
        Specialized function for one-dimensional arrays with unsigned integers.

    validate_arrayNx3
        Specialized function for Nx3 dimensional arrays.

    validate_data_range
        Specialized function for data ranges.


    Parameters
    ----------
    array : Number | Array
        Number or array to be validated, in any form that can be converted to
        a :class:`np.ndarray`. This includes lists, lists of tuples, tuples,
        tuples of tuples, tuples of lists and ndarrays.

    must_have_shape : ShapeLike | list[ShapeLike], optional
        :func:`Check <pyvista.core._validation.check.check_shape>`
        if the array has a specific shape. Specify a single shape
        or a ``list`` of any allowable shapes. If an integer, the array must
        be 1-dimensional with that length. Use a value of ``-1`` for any
        dimension where its size is allowed to vary. Use ``()`` to allow
        scalar values (i.e. 0-dimensional). Set to ``None`` if the array
        can have any shape (default).

    must_have_dtype : numpy.typing.DTypeLike | Sequence[numpy.typing.DTypeLike], optional
        :func:`Check <pyvista.core._validation.check.check_subdtype>`
        if the array's data-type has the given dtype. Specify a
        :class:`numpy.dtype` object or dtype-like base class which the
        array's data must be a subtype of. If a sequence, the array's data
        must be a subtype of at least one of the specified dtypes.

    must_have_length : int | Vector[int], optional
        :func:`Check <pyvista.core._validation.check.check_length>`
        if the array has the given length. If multiple values are given,
        the array's length must match one of the values.

        .. note ::

            The array's length is determined after reshaping the array
            (if ``reshape_to`` is not ``None``) and after broadcasting (if
            ``broadcast_to`` is not ``None``). Therefore, the specified length
            values should take the array's new shape into consideration if
            applicable.

    must_have_min_length : int, optional
        :func:`Check <pyvista.core._validation.check.check_length>`
        if the array's length is this value or greater. See note in
        ``must_have_length`` for details.

    must_have_max_length : int, optional
        :func:`Check <pyvista.core._validation.check.check_length>`
        if the array' length is this value or less. See note in
        ``must_have_length`` for details.

    must_be_nonnegative : bool, default: False
        :func:`Check <pyvista.core._validation.check.check_nonnegative>`
        if all elements of the array are nonnegative.

    must_be_finite : bool, default: False
        :func:`Check <pyvista.core._validation.check.check_finite>`
        if all elements of the array are finite, i.e. not ``infinity``
        and not Not a Number (``NaN``).

    must_be_real : bool, default: True
        :func:`Check <pyvista.core._validation.check.check_real>`
        if the array has real numbers, i.e. its data type is integer or
        floating.

        .. warning::

            Setting this parameter to ``False`` can result in unexpected
            behavior and is not recommended. There is limited support
            for complex number and/or string arrays.

    must_be_integer : bool, default: False
        :func:`Check <pyvista.core._validation.check.check_integer>`
        if the array's values are integer-like (i.e. that
        ``np.all(arr, np.floor(arr))``).

        .. note::

            Arrays with a float values may pass this check. Set
            ``dtype_out=int`` if the output must have an integer dtype.

    must_be_sorted : bool | dict, default: False
        :func:`Check <pyvista.core._validation.check.check_sorted>`
        if the array's values are sorted. If ``True``, the check is
        performed with default parameters:

        * ``ascending=True``: the array must be sorted in ascending order
        * ``strict=False``: sequential elements with the same value are allowed
        * ``axis=-1``: the sorting is checked along the array's last axis

        To check for descending order, enforce strict ordering, or to check
        along a different axis, use a ``dict`` with keyword arguments that
        will be passed to :func:`Check <pyvista.core._validation.check.check_sorted>`.

    must_be_in_range : Vector[float], optional
        :func:`Check <pyvista.core._validation.check.check_range>`
        if the array's values are all within a specific range. Range
        must be a vector with two elements specifying the minimum and
        maximum data values allowed, respectively. By default, the range
        endpoints are inclusive, i.e. values must be >= minimum and <=
        maximum. Use ``strict_lower_bound`` and/or ``strict_upper_bound``
        to further restrict the allowable range.

        ..note ::

            Use infinity (``np.inf`` or ``float('inf')``) to check for open
            intervals, e.g.:

            * ``[-np.inf, upper]`` to check if values are less
              than (or equal to) ``upper``
            * ``[lower, np.inf]`` to check if values are greater
              than (or equal to) ``lower``

    strict_lower_bound : bool, default: False
        Enforce a strict lower bound for the range specified by
        ``must_be_in_range``, i.e. array values must be strictly greater
        than the specified minimum.

    strict_upper_bound : bool, default: False
        Enforce a strict upper bound for the range specified by
        ``must_be_in_range``, i.e. array values must be strictly less
        than the specified maximum.

    reshape_to : int | tuple[int, ...], optional
        Reshape the output array to a new shape with :func:`numpy.reshape`.
        The shape should be compatible with the original shape. If an
        integer, then the result will be a 1-D array of that length. One
        shape dimension can be -1.

    broadcast_to : int | tuple[int, ...], optional
        Broadcast the array with :func:`numpy.broadcast_to` to a
        read-only view with the specified shape. Broadcasting is done
        after reshaping (if ``reshape_to`` is not ``None``).

    dtype_out : numpy.typing.DTypeLike, optional
        Set the data-type of the returned array. By default, the
        dtype is inferred from the input data. If ``dtype_out`` differs
        from the array's dtype, a copy of the array is made. The dtype
        of the array is set after any ``must_be_real`` or ``must_have_dtype``
        checks are made.

        .. warning::

            Setting this to a NumPy dtype (e.g. ``np.float64``) will implicitly
            set ``return_type`` to ``numpy``. Set to ``float``, ``int``, or
            ``bool`` to avoid this behavior.

        .. warning::

            Array validation can fail or result in silent integer overflow
            if ``dtype_out`` is integral and the input has infinity values.
            Consider setting ``must_be_finite=True`` for these cases.

    return_type : str | type, optional
        Control the return type of the array. Must be one of:

        * ``"numpy"`` or ``np.ndarray``
        * ``"list"`` or ``list``
        * ``"tuple"`` or ``tuple``

        .. note::

            For scalar inputs, setting the output type to ``list`` or
            ``tuple`` will return a scalar, and not an actual ``list``
            or ``tuple`` object.

    as_any : bool, default: True
        Allow subclasses of ``np.ndarray`` to pass through without
        making a copy. Has no effect if the input is not a NumPy array.

    copy : bool, default: False
        If ``True``, a copy of the array is returned. In some cases, a copy may be
        returned even if ``copy=False`` (e.g. to convert array type/dtype, reshape,
        etc.). In cases where the array is immutable (e.g. tuple) the returned array
        may not be a copy, even if ``copy=True``.

    get_flags : bool, default: False
        If ``True``, return a ``namedtuple`` of boolean flags with information about
        how the output may differ from the input. The flags returned are:

        - ``same_shape``:  ``True`` if the validated array has the same shape as the input.
          Always ``True`` if ``reshape_to`` and ``broadcast_to`` are ``None``.

        - ``same_dtype``: ``True`` if the validated array has the same dtype as the input.
          Always ``True`` if ``dtype_out`` is ``None``.

        - ``same_type``: ``True`` if the validated array has the same type as the input.
          Always ``True`` if ``return_type`` is ``None``.

        - ``same_object``: ``True`` if the validated array is the same object as the input.
          May be ``True`` or ``False`` based on a number of factors.

    name : str, default: "Array"
        Variable name to use in the error messages if any of the
        validation checks fail.

    Returns
    -------
    Number | Array
        Validated array of the same type and dtype as the input.
        If ``return_type`` is not ``None``, the returned array has the specified type.
        If ``dtype_out`` is not ``None``, the returned array has the specified dtype.
        See function description for more details.

    Examples
    --------
    Validate a one-dimensional array has at least length two, is
    monotonically increasing (i.e. has strict ascending order), and
    is within some range.

    >>> from pyvista import _validation
    >>> array_in = (1, 2, 3, 5, 8, 13)
    >>> rng = (0, 20)
    >>> _validation.validate_array(
    ...     array_in,
    ...     must_have_shape=(-1),
    ...     must_have_min_length=2,
    ...     must_be_sorted=dict(strict=True),
    ...     must_be_in_range=rng,
    ... )
    (1, 2, 3, 5, 8, 13)

    """
    type_in = type(array)
    id_in = id(array)
    wrapped = _ArrayLikeWrapper(array)

    # Check dtype
    if must_be_real:
        check_real(wrapped(), name=name)
    if must_have_dtype is not None:
        check_subdtype(wrapped(), base_dtype=must_have_dtype, name=name)

    # Validate dtype_out
    if dtype_out not in (None, float, int, bool):
        # Must be a numpy dtype
        check_subdtype(dtype_out, base_dtype=np.generic, name='dtype_out')
        if return_type is None:
            # Always return numpy array for numpy dtypes
            return_type = 'numpy'
        elif return_type in (tuple, list, 'tuple', 'list'):
            raise ValueError(
                f"Return type {return_type} is not compatible with dtype_out={dtype_out}.\n"
                f"A list or tuple can only be returned if dtype_out is float, int, or bool."
            )

    # Check if re-casting is needed in case subclasses are not allowed
    rewrap_numpy = as_any is False and type(wrapped._array) is not np.ndarray

    # Check if built-in types need to be cast to numpy in case broadcasting
    # or reshaping is needed (these are only supported for numpy arrays)
    do_reshape = reshape_to is not None and wrapped.shape != reshape_to
    do_broadcast = broadcast_to is not None and wrapped.shape != broadcast_to
    is_builtin = isinstance(wrapped, _BuiltinWrapper)
    rewrap_builtin = is_builtin and (do_reshape or do_broadcast)

    if rewrap_numpy or rewrap_builtin or return_type in ("numpy", np.ndarray):
        wrapped = (
            _ArrayLikeWrapper(np.asanyarray(array))
            if as_any
            else _ArrayLikeWrapper(np.asarray(array))
        )

    shape_in = wrapped.shape
    dtype_in = wrapped.dtype

    # Check shape
    if must_have_shape is not None:
        check_shape(wrapped(), shape=must_have_shape, name=name)

    # Do reshape _after_ checking shape to prevent unexpected reshaping
    if do_reshape or do_broadcast:
        wrapped._array = np.reshape(wrapped._array, reshape_to)  # type: ignore[arg-type]

    if do_broadcast:
        wrapped._array = np.broadcast_to(wrapped._array, broadcast_to, subok=True)  # type: ignore[arg-type]

    # Check length _after_ reshaping otherwise length may be wrong
    if (
        must_have_length is not None
        or must_have_min_length is not None
        or must_have_max_length is not None
    ):
        check_length(
            wrapped(),
            exact_length=must_have_length,
            min_length=must_have_min_length,
            max_length=must_have_max_length,
            allow_scalar=True,
            name=name,
        )

    # Check data values
    if must_be_nonnegative:
        check_nonnegative(wrapped(), name=name)
    # Check finite before setting dtype since dtype change can fail with inf
    if must_be_finite:
        check_finite(wrapped(), name=name)
    if must_be_integer:
        check_integer(wrapped(), strict=False, name=name)
    if must_be_in_range is not None:
        check_range(
            wrapped(),
            must_be_in_range,
            strict_lower=strict_lower_bound,
            strict_upper=strict_upper_bound,
            name=name,
        )
    if must_be_sorted:
        if isinstance(must_be_sorted, dict):
            check_sorted(wrapped(), **must_be_sorted, name=name)  # type: ignore[arg-type]
        else:
            check_sorted(wrapped(), name=name)

    # Set dtype
    if dtype_out is not None:
        try:
            wrapped.change_dtype(dtype_out)
        except OverflowError as e:
            if 'cannot convert float infinity to integer' in repr(e):
                raise TypeError(
                    f"Cannot change dtype of {name} from {wrapped.dtype} to {dtype_out}.\n"
                    f"Float infinity cannot be converted to integer."
                )
    # Cast array to desired output
    if return_type is None:
        if is_builtin:
            if isinstance(array, tuple):
                return_type = tuple
            else:
                # to-list will handle scalars and lists
                return_type = list
        else:
            return_type = np.ndarray

    def _get_flags(_wrapped, _out):
        return _ValidationFlags(
            same_shape=wrapped.shape == shape_in,
            same_dtype=np.dtype(dtype_out) == np.dtype(dtype_in),
            same_type=type(_out) is type_in,
            same_object=id(_out) == id_in,
        )

    if return_type in ("numpy", np.ndarray):
        out1 = wrapped.to_numpy(array, copy)
        return (out1, _get_flags(wrapped, out1)) if get_flags else out1
    elif return_type in ("list", list):
        out2 = wrapped.to_list(array, copy)
        return (out2, _get_flags(wrapped, out2)) if get_flags else out2
    elif return_type in ("tuple", tuple):
        out3 = wrapped.to_tuple(array, copy)
        return (out3, _get_flags(wrapped, out3)) if get_flags else out3
    else:
        # Invalid type, raise error with check
        check_contains(
            ["numpy", "list", "tuple", np.ndarray, list, tuple],
            must_contain=return_type,
            name='Return type',
        )


def validate_axes(
    *axes: Union[Matrix[float], Vector[float]],
    normalize: bool = True,
    must_be_orthogonal: bool = True,
    must_have_orientation: Optional[str] = 'right',
    name: str = "Axes",
) -> NumpyArray[float]:
    """Validate 3D axes vectors.

    By default, the axes are normalized and checked to ensure they are orthogonal and
    have a right-handed orientation.

    Parameters
    ----------
    *axes : Matrix[float] | Vector[float]
        Axes to be validated. Axes may be specified as a single array of row vectors
        or as separate arguments for each 3-element axis vector.
        If only two vectors are given and ``must_have_orientation`` is not ``None``,
        the third vector is automatically calculated as the cross-product of the
        two vectors such that the axes have the correct orientation.

    normalize : bool, default: True
        If ``True``, the axes vectors are individually normalized to each have a norm
        of 1.

    must_be_orthogonal : bool, default: True
        Check if the axes are orthogonal. If ``True``, the cross product between any
        two axes vectors must be parallel to the third.

    must_have_orientation : str, default: 'right'
        Check if the axes have a specific orientation. If ``right``, the
        cross-product of the first axis vector with the second must have a positive
        direction. If ``left``, the direction must be negative. If ``None``, the
        orientation is not checked.

    name : str, default: "Axes"
        Variable name to use in the error messages if any of the
        validation checks fail.

    Returns
    -------
    np.ndarray
        Validated 3x3 axes array of row vectors.

    Examples
    --------
    Validate an axes array.

    >>> import numpy as np
    >>> from pyvista import _validation
    >>> _validation.validate_axes(np.eye(3))
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

    Validate individual axes vectors as a 3x3 array.

    >>> _validation.validate_axes([1, 0, 0], [0, 1, 0], [0, 0, 1])
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

    Create a validated left-handed axes array from two vectors.

    >>> _validation.validate_axes(
    ...     [1, 0, 0], [0, 1, 0], must_have_orientation='left'
    ... )
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0., -1.]])

    """
    if must_have_orientation is not None:
        check_contains(
            ['right', 'left'], must_contain=must_have_orientation, name=f"{name} orientation"
        )

    # Validate number of args
    num_args = len(axes)
    if num_args not in (1, 2, 3):
        raise ValueError(
            "Incorrect number of axes arguments. Number of arguments must be either:\n"
            "\tOne arg (a single array with two or three vectors),"
            "\tTwo args (two vectors), or"
            "\tThree args (three vectors)."
        )

    # Validate axes array
    vector2: Optional[NumpyArray[float]] = None
    if num_args == 1:
        axes_array = validate_array(
            axes[0], must_have_shape=[(2, 3), (3, 3)], name=name, dtype_out=np.floating
        )
        vector0 = axes_array[0]
        vector1 = axes_array[1]
        if len(axes_array) == 3:
            vector2 = axes_array[2]
    else:
        vector0 = validate_array3(axes[0], name=f"{name} Vector[0]")
        vector1 = validate_array3(axes[1], name=f"{name} Vector[1]")
        if num_args == 3:
            vector2 = validate_array3(axes[2], name=f"{name} Vector[2]")

    if vector2 is None:
        if must_have_orientation is None:
            raise ValueError(
                f"{name} orientation must be specified when only two vectors are given."
            )
        elif must_have_orientation == 'right':
            vector2 = np.cross(vector0, vector1)
        else:
            vector2 = np.cross(vector1, vector0)
    axes_array = np.vstack((vector0, vector1, vector2))
    check_finite(axes_array, name=name)

    if np.any(np.all(np.isclose(axes_array, np.zeros(3)), axis=1)):
        raise ValueError(f"{name} cannot be zeros.")

    # Normalize axes for dot and cross product calcs
    axes_norm = axes_array / np.linalg.norm(axes_array, axis=1).reshape((3, 1))

    # Check non-parallel
    if np.isclose(np.dot(axes_norm[0], axes_norm[1]), 1) or np.isclose(
        np.dot(axes_norm[0], axes_norm[2]), 1
    ):
        raise ValueError(f"{name} cannot be parallel.")

    # Check orthogonality
    cross_0_1 = np.cross(axes_norm[0], axes_norm[1])
    cross_1_2 = np.cross(axes_norm[1], axes_norm[2])

    if must_be_orthogonal and not (
        (np.allclose(cross_0_1, axes_norm[2]) or np.allclose(cross_0_1, -axes_norm[2]))
        and (np.allclose(cross_1_2, axes_norm[0]) or np.allclose(cross_1_2, -axes_norm[0]))
    ):
        raise ValueError(f"{name} are not orthogonal.")

    # Check orientation
    if must_have_orientation:
        dot = np.dot(cross_0_1, axes_norm[2])
        if must_have_orientation == 'right' and dot < 0:
            raise ValueError(f"{name} do not have a right-handed orientation.")
        if must_have_orientation == 'left' and dot > 0:
            raise ValueError(f"{name} do not have a left-handed orientation.")

    return axes_norm if normalize else axes_array


def validate_transform4x4(transform: TransformLike, /, *, name="Transform"):
    """Validate transform-like input as a 4x4 ndarray.

    Parameters
    ----------
    transform : Matrix[float] | vtkTransform | vtkMatrix4x4 | vtkMatrix3x3
        Transformation matrix as a 3x3 or 4x4 array, 3x3 or 4x4 vtkMatrix,
        or as a vtkTransform.

    name : str, default: "Transform"
        Variable name to use in the error messages if any of the
        _validation checks fail.

    Returns
    -------
    np.ndarray
        Validated 4x4 transformation matrix.

    See Also
    --------
    validate_transform3x3
        Similar function for 3x3 transforms.

    validate_array
        Generic array _validation function.

    """
    array = np.eye(4)  # initialize
    if isinstance(transform, _vtk.vtkMatrix4x4):
        array = _array_from_vtkmatrix(transform, shape=(4, 4))
    elif isinstance(transform, _vtk.vtkMatrix3x3):
        array[:3, :3] = _array_from_vtkmatrix(transform, shape=(3, 3))
    elif isinstance(transform, _vtk.vtkTransform):
        array = _array_from_vtkmatrix(transform.GetMatrix(), shape=(4, 4))
    else:
        try:
            valid_array = validate_array(
                transform,
                must_have_shape=[(3, 3), (4, 4)],
                must_be_finite=True,
                name=name,
                return_type='numpy',
            )
            # reveal_type(valid_array)
            if valid_array.shape == (3, 3):
                array[:3, :3] = valid_array
            else:
                array = valid_array
        except ValueError:
            raise TypeError(
                'Input transform must be one of:\n'
                '\tvtkMatrix4x4\n'
                '\tvtkMatrix3x3\n'
                '\tvtkTransform\n'
                '\t4x4 np.ndarray\n'
                '\t3x3 np.ndarray\n'
            )

    return array


def validate_transform3x3(
    transform: Union[Matrix[float], _vtk.vtkMatrix3x3], /, *, name="Transform"
):
    """Validate transform-like input as a 3x3 ndarray.

    Parameters
    ----------
    transform : Matrix[float] | vtkMatrix3x3
        Transformation matrix as a 3x3 array or vtkMatrix3x3.

    name : str, default: "Transform"
        Variable name to use in the error messages if any of the
        _validation checks fail.

    Returns
    -------
    np.ndarray
        Validated 3x3 transformation matrix.

    See Also
    --------
    validate_transform4x4
        Similar function for 4x4 transforms.

    validate_array
        Generic array _validation function.

    """
    array = np.eye(3)  # initialize
    if isinstance(transform, _vtk.vtkMatrix3x3):
        array[:3, :3] = _array_from_vtkmatrix(transform, shape=(3, 3))
    else:
        try:
            array = validate_array(
                transform, must_have_shape=(3, 3), name=name, return_type="numpy"
            )
        except ValueError:
            raise TypeError(
                'Input transform must be one of:\n' '\tvtkMatrix3x3\n' '\t3x3 np.ndarray\n'
            )
    return array


def _array_from_vtkmatrix(
    matrix: Union[_vtk.vtkMatrix3x3, _vtk.vtkMatrix4x4],
    shape: Union[Tuple[Literal[3], Literal[3]], Tuple[Literal[4], Literal[4]]],
) -> NumpyArray[float]:
    """Convert a vtk matrix to an array."""
    array = np.zeros(shape)
    for i, j in product(range(shape[0]), range(shape[1])):
        array[i, j] = matrix.GetElement(i, j)
    return array


def validate_number(
    num: Union[_NumberType, Vector[_NumberType]], /, *, reshape=True, **kwargs
) -> _NumberType:
    """Validate a real, finite scalar number.

    By default, the number is checked to ensure it:

    * is scalar or is an array with one element
    * is a real number
    * is finite

    Parameters
    ----------
    num : float | int | Vector[float] | Vector[int]
        Number to validate.

    reshape : bool, default: True
        If ``True``, 1D arrays with 1 element are considered valid input
        and are reshaped to be 0-dimensional.

    **kwargs : dict, optional
        Additional keyword arguments passed to :func:`~validate_array`.

    Returns
    -------
    int | float
        Validated number.

    See Also
    --------
    validate_array
        Generic array validation function.

    Examples
    --------
    Validate a number.

    >>> from pyvista import _validation
    >>> _validation.validate_number(1)
    1

    1D arrays are automatically reshaped.

    >>> _validation.validate_number([42.0])
    42.0

    Additional checks can be added as needed.

    >>> _validation.validate_number(
    ...     10, must_be_in_range=[0, 10], must_be_integer=True
    ... )
    10

    """
    kwargs.setdefault('name', 'Number')
    kwargs.setdefault('must_be_finite', True)
    kwargs.setdefault('must_be_real', True)

    shape: Union[_ShapeLike, List[_ShapeLike]]
    if reshape:
        shape = [(), (1,)]
        _set_default_kwarg_mandatory(kwargs, 'reshape_to', ())
    else:
        shape = ()
    _set_default_kwarg_mandatory(kwargs, 'must_have_shape', shape)

    return validate_array(num, **kwargs)


def validate_data_range(rng: Vector[_NumberType], /, **kwargs):
    """Validate a data range.

    By default, the data range is checked to ensure:

    * it has two values
    * it has real numbers
    * the lower bound is not more than the upper bound

    Parameters
    ----------
    rng : Vector[float]
        Range to validate in the form ``(lower_bound, upper_bound)``.

    **kwargs : dict, optional
        Additional keyword arguments passed to :func:`~validate_array`.

    Returns
    -------
    tuple
        Validated range as ``(lower_bound, upper_bound)``.

    See Also
    --------
    validate_array
        Generic array validation function.

    Examples
    --------
    Validate a data range.

    >>> from pyvista import _validation
    >>> _validation.validate_data_range([-5, 5.0])
    [-5, 5.0]

    Add additional constraints if needed, e.g. to ensure the output
    only contains floats.

    >>> _validation.validate_data_range([-5, 5.0], dtype_out=float)
    [-5.0, 5.0]

    """
    kwargs.setdefault('name', 'Data Range')
    _set_default_kwarg_mandatory(kwargs, 'must_have_shape', 2)
    _set_default_kwarg_mandatory(kwargs, 'must_be_sorted', True)
    return validate_array(rng, **kwargs)


def validate_arrayNx3(
    array: Union[Matrix[_NumberType], Vector[_NumberType]], /, *, reshape=True, **kwargs
) -> NumpyArray[_NumberType]:
    """Validate an array is numeric and has shape Nx3.

    The array is checked to ensure its input values:

    * have shape ``(N, 3)`` or can be reshaped to ``(N, 3)``
    * are numeric

    The returned array is formatted so that its values:

    * have shape ``(N, 3)``.

    Parameters
    ----------
    array : Vector[float] | Matrix[float]
        1D or 2D array to validate.

    reshape : bool, default: True
        If ``True``, 1D arrays with 3 elements are considered valid
        input and are reshaped to ``(1, 3)`` to ensure the output is
        two-dimensional.

    **kwargs : dict, optional
        Additional keyword arguments passed to :func:`~validate_array`.

    Returns
    -------
    np.ndarray
        Validated array with shape ``(N, 3)``.

    See Also
    --------
    validate_arrayN
        Similar function for one-dimensional arrays.

    validate_array
        Generic array validation function.

    Examples
    --------
    Validate an Nx3 array.

    >>> from pyvista import _validation
    >>> _validation.validate_arrayNx3(((1, 2, 3), (4, 5, 6)))
    ((1, 2, 3), (4, 5, 6))

    One-dimensional 3-element arrays are automatically reshaped to 2D.

    >>> _validation.validate_arrayNx3([1, 2, 3])
    [[1, 2, 3]]

    Add additional constraints.

    >>> _validation.validate_arrayNx3(
    ...     ((1, 2, 3), (4, 5, 6)), must_be_in_range=[0, 10]
    ... )
    ((1, 2, 3), (4, 5, 6))

    """
    shape: Union[_ShapeLike, List[_ShapeLike]]
    if reshape:
        shape = [3, (-1, 3)]
        _set_default_kwarg_mandatory(kwargs, 'reshape_to', (-1, 3))
    else:
        shape = (-1, 3)
    _set_default_kwarg_mandatory(kwargs, 'must_have_shape', shape)

    return validate_array(array, **kwargs)


def validate_arrayN(
    array: Union[_NumberType, Vector[_NumberType], Matrix[_NumberType]],
    /,
    *,
    reshape=True,
    **kwargs,
):
    """Validate a numeric 1D array.

    The array is checked to ensure its input values:

    * have shape ``(N,)`` or can be reshaped to ``(N,)``
    * are numeric

    The returned array is formatted so that its values:

    * have shape ``(N,)``

    Parameters
    ----------
    array : float | Vector[float] | Matrix[float]
        Array-like input to validate.

    reshape : bool, default: True
        If ``True``, 0-dimensional scalars are reshaped to ``(1,)`` and 2D
        vectors with shape ``(1, N)`` are reshaped to ``(N,)`` to ensure the
        output is consistently one-dimensional. Otherwise, all scalar and
        2D inputs are not considered valid.

    **kwargs : dict, optional
        Additional keyword arguments passed to :func:`~validate_array`.

    Returns
    -------
    np.ndarray
        Validated 1D array.

    See Also
    --------
    validate_arrayN_unsigned
        Similar function for non-negative integer arrays.

    validate_array
        Generic array validation function.

    Examples
    --------
    Validate a 1D array with four elements.

    >>> from pyvista import _validation
    >>> _validation.validate_arrayN((1, 2, 3, 4))
    (1, 2, 3, 4)

    Scalar 0-dimensional values are automatically reshaped to be 1D.

    >>> _validation.validate_arrayN(42.0)
    [42.0]

    2D arrays where the first dimension is unity are automatically
    reshaped to be 1D.

    >>> _validation.validate_arrayN([[1, 2]])
    [1, 2]

    Add additional constraints if needed.

    >>> _validation.validate_arrayN((1, 2, 3), must_have_length=3)
    (1, 2, 3)

    """
    shape: Union[_ShapeLike, List[_ShapeLike]]
    if reshape:
        shape = [(), (-1), (1, -1), (-1, 1)]
        _set_default_kwarg_mandatory(kwargs, 'reshape_to', (-1))
    else:
        shape = -1
    _set_default_kwarg_mandatory(kwargs, 'must_have_shape', shape)
    return validate_array(array, **kwargs)


def validate_arrayN_unsigned(
    array: Union[_NumberType, Vector[_NumberType], Matrix[_NumberType]],
    /,
    *,
    reshape=True,
    **kwargs,
):
    """Validate a numeric 1D array of non-negative (unsigned) integers.

    The array is checked to ensure its input values:

    * have shape ``(N,)`` or can be reshaped to ``(N,)``
    * are integer-like
    * are non-negative

    The returned array is formatted so that its values:

    * have shape ``(N,)``
    * have an integer data type

    Parameters
    ----------
    array : float | Vector[float] | Matrix[float]
        0D, 1D, or 2D array to validate.

    reshape : bool, default: True
        If ``True``, 0-dimensional scalars are reshaped to ``(1,)`` and 2D
        vectors with shape ``(1, N)`` are reshaped to ``(N,)`` to ensure the
        output is consistently one-dimensional. Otherwise, all scalar and
        2D inputs are not considered valid.

    **kwargs : dict, optional
        Additional keyword arguments passed to :func:`~validate_array`.

    Returns
    -------
    np.ndarray
        Validated 1D array with non-negative integers.

    See Also
    --------
    validate_arrayN
        Similar function for numeric one-dimensional arrays.

    validate_array
        Generic array validation function.

    Examples
    --------
    Validate a 1D array with four non-negative integer-like elements.

    >>> import numpy as np
    >>> from pyvista import _validation
    >>> array = _validation.validate_arrayN_unsigned((1.0, 2.0, 3.0, 4.0))
    >>> array
    (1, 2, 3, 4)

    Verify that the output data type is integral.

    >>> isinstance(array[0], int)
    True

    Scalar 0-dimensional values are automatically reshaped to be 1D.

    >>> _validation.validate_arrayN_unsigned(42)
    [42]

    2D arrays where the first dimension is unity are automatically
    reshaped to be 1D.

    >>> _validation.validate_arrayN_unsigned([[1, 2]])
    [1, 2]

    Add additional constraints if needed.

    >>> _validation.validate_arrayN_unsigned(
    ...     (1, 2, 3), must_be_in_range=[1, 3]
    ... )
    (1, 2, 3)

    """
    # Set default dtype out but allow overriding as long as the dtype
    # is also integral
    kwargs.setdefault('dtype_out', int)
    if kwargs['dtype_out'] is not int:
        check_subdtype(kwargs['dtype_out'], np.integer)

    # can overflow if not finite
    kwargs.setdefault('must_be_finite', True)

    _set_default_kwarg_mandatory(kwargs, 'must_be_integer', True)
    _set_default_kwarg_mandatory(kwargs, 'must_be_nonnegative', True)

    return validate_arrayN(array, reshape=reshape, **kwargs)


def validate_array3(
    array: Union[_NumberType, Vector[_NumberType], Matrix[_NumberType]],
    /,
    *,
    reshape=True,
    broadcast=False,
    **kwargs,
):
    """Validate a numeric 1D array with 3 elements.

    The array is checked to ensure its input values:

    * have shape ``(3,)`` or can be reshaped to ``(3,)``
    * are numeric and real

    The returned array is formatted so that it has shape ``(3,)``.

    Parameters
    ----------
    array : float | Vector[float] | Matrix[float]
        Array to validate.

    reshape : bool, default: True
        If ``True``, 2D vectors with shape ``(1, 3)`` or ``(3, 1)`` are
        considered valid input, and are reshaped to ``(3,)`` to ensure
        the output is consistently one-dimensional.

    broadcast : bool, default: False
        If ``True``, scalar values or 1D arrays with a single element
        are considered valid input and the single value is broadcast to
        a length 3 array.

    **kwargs : dict, optional
        Additional keyword arguments passed to :func:`~validate_array`.

    Returns
    -------
    np.ndarray
        Validated 1D array with 3 elements.

    See Also
    --------
    validate_number
        Similar function for a single number.

    validate_arrayN
        Similar function for one-dimensional arrays.

    validate_array
        Generic array validation function.

    Examples
    --------
    Validate a 1D array with three elements.

    >>> from pyvista import _validation
    >>> _validation.validate_array3((1, 2, 3))
    (1, 2, 3)

    2D 3-element arrays are automatically reshaped to be 1D.

    >>> _validation.validate_array3([[1, 2, 3]])
    [1, 2, 3]

    Scalar 0-dimensional values can be automatically broadcast as
    a 3-element 1D array.

    >>> _validation.validate_array3(42.0, broadcast=True)
    [42.0, 42.0, 42.0]

    Add additional constraints if needed.

    >>> _validation.validate_array3((1, 2, 3), must_be_nonnegative=True)
    (1, 2, 3)

    """
    shape: List[tuple[int, ...]]
    shape = [(3,)]
    if reshape:
        shape.append((1, 3))
        shape.append((3, 1))
        _set_default_kwarg_mandatory(kwargs, 'reshape_to', (-1))
    if broadcast:
        shape.append(())  # allow 0D scalars
        shape.append((1,))  # 1D 1-element vectors
        _set_default_kwarg_mandatory(kwargs, 'broadcast_to', (3,))
    _set_default_kwarg_mandatory(kwargs, 'must_have_shape', shape)

    return validate_array(array, **kwargs)


def _set_default_kwarg_mandatory(kwargs: Dict[str, Any], key: str, default: Any):
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
