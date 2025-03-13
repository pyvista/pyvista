"""Functions that validate input and return a standard representation.

.. versionadded:: 0.43.0

A ``validate`` function typically:

* Uses :py:mod:`~pyvista.core._validation.check` functions to
  check the type and/or value of input arguments.
* Applies (optional) constraints, e.g. input or output must have a
  specific length, shape, type, data-type, etc.
* Accepts many different input types or values and standardizes the
  output as a single representation with known properties.

"""

from __future__ import annotations

import inspect
from itertools import product
import reprlib
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import NamedTuple
from typing import Optional
from typing import TypedDict
from typing import TypeVar
from typing import Union
from typing import cast
from typing import overload

import numpy as np

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

from pyvista.core import _vtk_core as _vtk
from pyvista.core._typing_core._array_like import _NumberType
from pyvista.core._typing_core._array_like import _NumberUnion
from pyvista.core._validation._cast_array import _cast_to_numpy
from pyvista.core._validation._cast_array import _cast_to_tuple
from pyvista.core._validation.check import _ShapeLike
from pyvista.core._validation.check import check_contains
from pyvista.core._validation.check import check_finite
from pyvista.core._validation.check import check_integer
from pyvista.core._validation.check import check_length
from pyvista.core._validation.check import check_ndim
from pyvista.core._validation.check import check_nonnegative
from pyvista.core._validation.check import check_range
from pyvista.core._validation.check import check_real
from pyvista.core._validation.check import check_shape
from pyvista.core._validation.check import check_sorted
from pyvista.core._validation.check import check_string
from pyvista.core._validation.check import check_subdtype

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from pyvista.core._typing_core import MatrixLike
    from pyvista.core._typing_core import NumberType
    from pyvista.core._typing_core import NumpyArray
    from pyvista.core._typing_core import RotationLike
    from pyvista.core._typing_core import TransformLike
    from pyvista.core._typing_core import VectorLike
    from pyvista.core._typing_core._aliases import _ArrayLikeOrScalar
    from pyvista.core._typing_core._array_like import _FiniteNestedList
    from pyvista.core._typing_core._array_like import _FiniteNestedTuple
    from pyvista.plotting._typing import ColorLike
    from pyvista.plotting.colors import Color

    from .check import _ShapeLike


class _ValidationFlags(NamedTuple):
    same_shape: bool
    same_dtype: bool
    same_type: bool
    same_object: bool


_FloatType = TypeVar('_FloatType', bound=float)  # noqa: PYI018

_NumpyReturnType = Union[
    Literal['numpy'],
    type[np.ndarray],  # type: ignore[type-arg]
]
_ListReturnType = Union[
    Literal['list'],
    type[list],  # type: ignore[type-arg]
]
_TupleReturnType = Union[
    Literal['tuple'],
    type[tuple],  # type: ignore[type-arg]
]
_ArrayReturnType = Union[_NumpyReturnType, _ListReturnType, _TupleReturnType]


class _TypedKwargs(TypedDict, total=False):
    must_have_shape: Optional[Union[_ShapeLike, list[_ShapeLike]]]
    must_have_ndim: Optional[int]
    must_have_dtype: Optional[_NumberUnion]
    must_have_length: Optional[Union[int, VectorLike[int]]]
    must_have_min_length: Optional[int]
    must_have_max_length: Optional[int]
    must_be_nonnegative: bool
    must_be_finite: bool
    must_be_real: bool
    must_be_integer: bool
    must_be_sorted: Union[bool, dict[str, Union[bool, int]]]
    must_be_in_range: Optional[VectorLike[float]]
    strict_lower_bound: bool
    strict_upper_bound: bool
    as_any: bool
    copy: bool
    get_flags: bool
    name: str


# Define overloads for validate_array
# Overloads are listed in a similar order as the runtime isinstance checks performed
# by the array wrappers, and generally go from most specific to least specific:
#       scalars -> flat or nested lists and tuples -> numpy arrays -> general array-like
# See https://mypy.readthedocs.io/en/stable/more_types.html#function-overloading
#
# """SCALAR OVERLOADS"""
# T -> T
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: NumberType,
    /,
    *,
    dtype_out: None = None,
    return_type: Optional[Union[_TupleReturnType, _ListReturnType]] = ...,
    reshape_to: Optional[tuple[()]] = ...,
    broadcast_to: Optional[tuple[()]] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> NumberType: ...


# T1 -> T2
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: NumberType,
    /,
    *,
    dtype_out: type[_NumberType],
    return_type: Optional[Union[_TupleReturnType, _ListReturnType]] = ...,
    reshape_to: Optional[tuple[()]] = ...,
    broadcast_to: Optional[tuple[()]] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> _NumberType: ...


# T -> NDArray[T]
@overload
def validate_array(  # numpydoc ignore=GL08
    array: NumberType,
    /,
    *,
    dtype_out: None = None,
    return_type: _NumpyReturnType,
    reshape_to: Optional[tuple[()]] = ...,
    broadcast_to: Optional[tuple[()]] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> NumpyArray[NumberType]: ...


# T1 -> NDArray[T2]
@overload
def validate_array(  # numpydoc ignore=GL08
    array: NumberType,
    /,
    *,
    dtype_out: type[_NumberType],
    return_type: _NumpyReturnType,
    reshape_to: Optional[tuple[()]] = ...,
    broadcast_to: Optional[tuple[()]] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> NumpyArray[_NumberType]: ...


# """LIST OVERLOADS"""
# list[list[T]] -> list[list[T]]
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: list[list[NumberType]],
    /,
    *,
    dtype_out: None = None,
    return_type: Optional[_ListReturnType] = ...,
    reshape_to: Optional[tuple[int, int]] = ...,
    broadcast_to: Optional[tuple[int, int]] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> list[list[NumberType]]: ...


# list[list[T1]] -> list[list[T2]]
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: list[list[NumberType]],
    /,
    *,
    dtype_out: type[_NumberType],
    return_type: Optional[_ListReturnType] = ...,
    reshape_to: Optional[tuple[int, int]] = ...,
    broadcast_to: Optional[tuple[int, int]] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> list[list[_NumberType]]: ...


# list[list[T]] -> tuple[tuple[T]]
@overload
def validate_array(  # numpydoc ignore=GL08
    array: list[list[NumberType]],
    /,
    *,
    dtype_out: None = None,
    return_type: _TupleReturnType,
    reshape_to: Optional[tuple[int, int]] = ...,
    broadcast_to: Optional[tuple[int, int]] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> tuple[tuple[NumberType]]: ...


# list[list[T1]] -> tuple[tuple[T2]]
@overload
def validate_array(  # numpydoc ignore=GL08
    array: list[list[NumberType]],
    /,
    *,
    dtype_out: type[_NumberType],
    return_type: _TupleReturnType,
    reshape_to: Optional[tuple[int, int]] = ...,
    broadcast_to: Optional[tuple[int, int]] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> tuple[tuple[_NumberType]]: ...


# list[T] -> list[T]
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: list[NumberType],
    /,
    *,
    dtype_out: None = None,
    return_type: Optional[_ListReturnType] = ...,
    reshape_to: Optional[Union[int, tuple[int]]] = ...,
    broadcast_to: Optional[Union[int, tuple[int]]] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> list[NumberType]: ...


# list[T1] -> list[T2]
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: list[NumberType],
    /,
    *,
    dtype_out: type[_NumberType],
    return_type: Optional[_ListReturnType] = ...,
    reshape_to: Optional[Union[int, tuple[int]]] = ...,
    broadcast_to: Optional[Union[int, tuple[int]]] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> list[_NumberType]: ...


# list[T] -> tuple[T]
@overload
def validate_array(  # numpydoc ignore=GL08
    array: list[NumberType],
    /,
    *,
    dtype_out: None = None,
    return_type: _TupleReturnType,
    reshape_to: Optional[Union[int, tuple[int]]] = ...,
    broadcast_to: Optional[Union[int, tuple[int]]] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> tuple[NumberType]: ...


# list[T1] -> tuple[T2]
@overload
def validate_array(  # numpydoc ignore=GL08
    array: list[NumberType],
    /,
    *,
    dtype_out: type[_NumberType],
    return_type: _TupleReturnType,
    reshape_to: Optional[Union[int, tuple[int]]] = ...,
    broadcast_to: Optional[Union[int, tuple[int]]] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> tuple[_NumberType]: ...


# FiniteNestedlist[T] -> FiniteNestedlist[T]
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: _FiniteNestedList[NumberType],
    /,
    *,
    dtype_out: None = None,
    return_type: Optional[_ListReturnType] = ...,
    reshape_to: Optional[_ShapeLike] = ...,
    broadcast_to: Optional[_ShapeLike] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> Union[NumberType, _FiniteNestedList[NumberType]]: ...


# FiniteNestedlist[T1] -> FiniteNestedlist[T2]
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: _FiniteNestedList[NumberType],
    /,
    *,
    dtype_out: type[_NumberType],
    return_type: Optional[_ListReturnType] = ...,
    reshape_to: Optional[_ShapeLike] = ...,
    broadcast_to: Optional[_ShapeLike] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> Union[_NumberType, _FiniteNestedList[_NumberType]]: ...


# """TUPLE OVERLOADS"""
# tuple[tuple[T]] -> tuple[tuple[T]]
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: tuple[tuple[NumberType]],
    /,
    *,
    dtype_out: None = None,
    return_type: Optional[_TupleReturnType] = ...,
    reshape_to: Optional[tuple[int, int]] = ...,
    broadcast_to: Optional[tuple[int, int]] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> tuple[tuple[NumberType]]: ...


# tuple[tuple[T]] -> list[list[T]]
@overload
def validate_array(  # numpydoc ignore=GL08
    array: tuple[tuple[NumberType]],
    /,
    *,
    dtype_out: None = None,
    return_type: _ListReturnType,
    reshape_to: Optional[tuple[int, int]] = ...,
    broadcast_to: Optional[tuple[int, int]] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> list[list[NumberType]]: ...


# tuple[tuple[T1]] -> tuple[tuple[T2]]
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: tuple[tuple[NumberType]],
    /,
    *,
    dtype_out: type[_NumberType],
    return_type: Optional[_TupleReturnType] = ...,
    reshape_to: Optional[tuple[int, int]] = ...,
    broadcast_to: Optional[tuple[int, int]] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> tuple[tuple[_NumberType]]: ...


# tuple[tuple[T1]] -> list[list[T2]]
@overload
def validate_array(  # numpydoc ignore=GL08
    array: tuple[tuple[NumberType, ...]],
    /,
    *,
    dtype_out: type[_NumberType],
    return_type: _ListReturnType,
    reshape_to: Optional[tuple[int, int]] = ...,
    broadcast_to: Optional[tuple[int, int]] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> list[list[_NumberType]]: ...


# tuple[T] -> tuple[T]
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: tuple[NumberType, ...],
    /,
    *,
    dtype_out: None = None,
    return_type: Optional[_TupleReturnType] = ...,
    reshape_to: Optional[Union[int, tuple[int]]] = ...,
    broadcast_to: Optional[Union[int, tuple[int]]] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> tuple[NumberType]: ...


# tuple[T] -> list[T]
@overload
def validate_array(  # numpydoc ignore=GL08
    array: tuple[NumberType],
    /,
    *,
    dtype_out: None = None,
    return_type: _ListReturnType,
    reshape_to: Optional[Union[int, tuple[int]]] = ...,
    broadcast_to: Optional[Union[int, tuple[int]]] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> list[NumberType]: ...


# tuple[T1] -> tuple[T2]
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: tuple[NumberType, ...],
    /,
    *,
    dtype_out: type[_NumberType],
    return_type: Optional[_TupleReturnType] = ...,
    reshape_to: Optional[Union[int, tuple[int]]] = ...,
    broadcast_to: Optional[Union[int, tuple[int]]] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> tuple[_NumberType]: ...


# tuple[T1] -> list[T2]
@overload
def validate_array(  # numpydoc ignore=GL08
    array: tuple[NumberType, ...],
    /,
    *,
    dtype_out: type[_NumberType],
    return_type: _ListReturnType,
    reshape_to: Optional[Union[int, tuple[int]]] = ...,
    broadcast_to: Optional[Union[int, tuple[int]]] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> list[_NumberType]: ...


# FiniteNestedtuple[T] -> FiniteNestedtuple[T]
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: _FiniteNestedTuple[NumberType],
    /,
    *,
    dtype_out: None = None,
    return_type: Optional[_TupleReturnType] = ...,
    reshape_to: Optional[_ShapeLike] = ...,
    broadcast_to: Optional[_ShapeLike] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> Union[NumberType, _FiniteNestedTuple[NumberType]]: ...


# FiniteNestedtuple[T1] -> FiniteNestedtuple[T2]
@overload
def validate_array(  # type: ignore[overload-overlap]  # numpydoc ignore=GL08
    array: _FiniteNestedTuple[NumberType],
    /,
    *,
    dtype_out: type[_NumberType],
    return_type: Optional[_TupleReturnType] = ...,
    reshape_to: Optional[_ShapeLike] = ...,
    broadcast_to: Optional[_ShapeLike] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> Union[_NumberType, _FiniteNestedTuple[_NumberType]]: ...


# """NUMPY OVERLOADS"""
# NDArray[T] -> NDArray[T]
@overload
def validate_array(  # numpydoc ignore=GL08
    array: NumpyArray[NumberType],
    /,
    *,
    dtype_out: None = None,
    return_type: Optional[_NumpyReturnType] = ...,
    reshape_to: Optional[_ShapeLike] = ...,
    broadcast_to: Optional[_ShapeLike] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> NumpyArray[NumberType]: ...


# NDArray[T1] -> NDArray[T2]
@overload
def validate_array(  # numpydoc ignore=GL08
    array: NumpyArray[NumberType],
    /,
    *,
    dtype_out: type[_NumberType],
    return_type: Optional[_NumpyReturnType] = ...,
    reshape_to: Optional[_ShapeLike] = ...,
    broadcast_to: Optional[_ShapeLike] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> NumpyArray[_NumberType]: ...


# NDArray[T] -> FiniteNestedlist[T]
@overload
def validate_array(  # numpydoc ignore=GL08
    array: NumpyArray[NumberType],
    /,
    *,
    dtype_out: None = None,
    return_type: _ListReturnType,
    reshape_to: Optional[_ShapeLike] = ...,
    broadcast_to: Optional[_ShapeLike] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> Union[NumberType, _FiniteNestedList[NumberType]]: ...


# NDArray[T1] -> FiniteNestedlist[T2]
@overload
def validate_array(  # numpydoc ignore=GL08
    array: NumpyArray[NumberType],
    /,
    *,
    dtype_out: type[_NumberType],
    return_type: _ListReturnType,
    reshape_to: Optional[_ShapeLike] = ...,
    broadcast_to: Optional[_ShapeLike] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> Union[_NumberType, _FiniteNestedList[_NumberType]]: ...


# NDArray[T] -> Nestedtuple[T]
@overload
def validate_array(  # numpydoc ignore=GL08
    array: NumpyArray[NumberType],
    /,
    *,
    dtype_out: None = None,
    return_type: _TupleReturnType,
    reshape_to: Optional[_ShapeLike] = ...,
    broadcast_to: Optional[_ShapeLike] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> Union[NumberType, _FiniteNestedTuple[NumberType]]: ...


# NDArray[T1] -> FiniteNestedtuple[T2]
@overload
def validate_array(  # numpydoc ignore=GL08
    array: NumpyArray[NumberType],
    /,
    *,
    dtype_out: type[_NumberType],
    return_type: _TupleReturnType,
    reshape_to: Optional[_ShapeLike] = ...,
    broadcast_to: Optional[_ShapeLike] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> Union[_NumberType, _FiniteNestedTuple[_NumberType]]: ...


# """ARRAY-LIKE OVERLOADS"""
# These are general catch-all cases for anything not overloaded explicitly
# ArrayLike[T] -> FiniteNestedlist[T]
@overload
def validate_array(  # numpydoc ignore=GL08
    array: _ArrayLikeOrScalar[NumberType],
    /,
    *,
    dtype_out: None = None,
    return_type: _ListReturnType,
    reshape_to: Optional[_ShapeLike] = ...,
    broadcast_to: Optional[_ShapeLike] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> Union[NumberType, _FiniteNestedList[NumberType]]: ...


# ArrayLike[T1] -> FiniteNestedlist[T2]
@overload
def validate_array(  # numpydoc ignore=GL08
    array: _ArrayLikeOrScalar[NumberType],
    /,
    *,
    dtype_out: type[_NumberType],
    return_type: _ListReturnType,
    reshape_to: Optional[_ShapeLike] = ...,
    broadcast_to: Optional[_ShapeLike] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> Union[_NumberType, _FiniteNestedList[_NumberType]]: ...


# ArrayLike[T] -> FiniteNestedtuple[T]
@overload
def validate_array(  # numpydoc ignore=GL08
    array: _ArrayLikeOrScalar[NumberType],
    /,
    *,
    dtype_out: None = None,
    return_type: _TupleReturnType,
    reshape_to: Optional[_ShapeLike] = ...,
    broadcast_to: Optional[_ShapeLike] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> Union[NumberType, _FiniteNestedTuple[NumberType]]: ...


# ArrayLike[T1] -> FiniteNestedtuple[T2]
@overload
def validate_array(  # numpydoc ignore=GL08
    array: _ArrayLikeOrScalar[NumberType],
    /,
    *,
    dtype_out: type[_NumberType],
    return_type: _TupleReturnType,
    reshape_to: Optional[_ShapeLike] = ...,
    broadcast_to: Optional[_ShapeLike] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> Union[_NumberType, _FiniteNestedTuple[_NumberType]]: ...


# ArrayLike[T] -> NDArray[T]
@overload
def validate_array(  # numpydoc ignore=GL08
    array: _ArrayLikeOrScalar[NumberType],
    /,
    *,
    dtype_out: None = None,
    return_type: Optional[_NumpyReturnType] = ...,
    reshape_to: Optional[_ShapeLike] = ...,
    broadcast_to: Optional[_ShapeLike] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> NumpyArray[NumberType]: ...


# ArrayLike[T1] -> NDArray[T2]
@overload
def validate_array(  # numpydoc ignore=GL08
    array: _ArrayLikeOrScalar[NumberType],
    /,
    *,
    dtype_out: type[_NumberType],
    return_type: Optional[_NumpyReturnType] = ...,
    reshape_to: Optional[_ShapeLike] = ...,
    broadcast_to: Optional[_ShapeLike] = ...,
    **kwargs: Unpack[_TypedKwargs],
) -> NumpyArray[_NumberType]: ...
def validate_array(
    array: _ArrayLikeOrScalar[NumberType],
    /,
    *,
    must_have_shape: _ShapeLike | list[_ShapeLike] | None = None,
    must_have_ndim: int | VectorLike[int] | None = None,
    must_have_dtype: npt.DTypeLike | None = None,
    must_have_length: int | VectorLike[int] | None = None,
    must_have_min_length: int | None = None,
    must_have_max_length: int | None = None,
    must_be_nonnegative: bool = False,
    must_be_finite: bool = False,
    must_be_real: bool = True,
    must_be_integer: bool = False,
    must_be_sorted: bool | dict[str, int | bool] = False,
    must_be_in_range: VectorLike[float] | None = None,
    strict_lower_bound: bool = False,
    strict_upper_bound: bool = False,
    reshape_to: int | tuple[int, ...] | None = None,
    broadcast_to: int | tuple[int, ...] | None = None,
    dtype_out: npt.DTypeLike = None,
    as_any: bool = True,
    copy: bool = False,
    get_flags: bool = False,
    to_list: bool = False,
    to_tuple: bool = False,
    name: str = 'Array',
):
    """Check and validate a numeric array meets specific requirements.

    Validate an array to ensure it is numeric, has a specific shape,
    data-type, and/or has values that meet specific requirements such as
    being sorted, having integer values, or is finite. The array can
    optionally be reshaped or broadcast, and the return type of
    the array can be explicitly set to standardize its representation.

    By default, this function is generic and returns an array with the
    same type and dtype as the input array, i.e. ``Array[T] -> Array[T]``
    It is specifically designed to return the following array types as-is
    without copying its data where possible:

    - Scalars: ``T`` -> ``T``
    - Lists: ``list[T]`` -> ``list[T]``
    - Nested lists: ``list[list[T]]`` -> ``list[list[T]]``
    - Tuples: ``tuple[T]`` -> ``tuple[T]``
    - Nested tuples: ``tuple[tuple[T]]`` -> ``tuple[tuple[T]]``
    - NumPy arrays: ``NDArray[T]`` -> ``NDArray[T]``

    All other inputs (e.g. ``range`` objects) may first be copied to a
    NumPy array for processing internally. NumPy protocol arrays (e.g.
    ``pandas`` arrays) are not copied but are returned as a NumPy array.

    Optionally, use ``return_type`` and/or ``dtype_out`` for non-generic
    behavior to ensure the output array has a consistent type.

    .. warning::

        This function is primarily designed to work with homogeneous
        numeric arrays with a regular shape. Any other array-like
        inputs (e.g. structured arrays, string arrays) are not
        supported.

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

    must_have_shape : int | tuple[int, ...] | list[int, tuple[int, ...]], optional
        :func:`Check <pyvista.core._validation.check.check_shape>`
        if the array has a specific shape. Specify a single shape
        or a list of any allowable shapes. If an integer, the array must
        be 1-dimensional with that length. Use a value of ``-1`` for any
        dimension where its size is allowed to vary. Use ``()`` to allow
        scalar values (i.e. 0-dimensional). Set to ``None`` if the array
        can have any shape (default).

    must_have_ndim : int | VectorLike[int], optional
        :func:`Check <pyvista.core._validation.check.check_ndim>` if
        the array has the specified number of dimension(s). Specify a
        single dimension or a sequence of allowable dimensions. If a
        sequence, the array must have at least one of the specified
        number of dimensions.

    must_have_dtype : DTypeLike | list[DTypeLike, ...], optional
        :func:`Check <pyvista.core._validation.check.check_subdtype>`
        if the array's data-type has the given dtype. Specify a
        :class:`numpy.dtype` object or dtype-like base class which the
        array's data must be a subtype of. If a sequence, the array's data
        must be a subtype of at least one of the specified dtypes.

    must_have_length : int | VectorLike[int], optional
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

            This check does not require the input dtype to be integers,
            i.e. floats are allowed. Set ``must_have_dtype=int`` if the
            input is required to be integers or ``dtype_out=int`` to
            cast the output to integers.

    must_be_nonnegative : bool, default: False
        :func:`Check <pyvista.core._validation.check.check_nonnegative>`
        if all elements of the array are nonnegative. Consider also
        setting ``dtype_out``, e.g. to ensure the output is an unsigned
        integer type.

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

    must_be_in_range : VectorLike[float], optional
        :func:`Check <pyvista.core._validation.check.check_range>`
        if the array's values are all within a specific range. Range
        must be a vector with two elements specifying the minimum and
        maximum data values allowed, respectively. By default, the range
        endpoints are inclusive, i.e. values must be >= minimum and <=
        maximum. Use ``strict_lower_bound`` and/or ``strict_upper_bound``
        to further restrict the allowable range.

        .. note::

            Use infinity (``np.inf`` or ``float('inf')``) to specify an
            unlimited bound, e.g.:

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
        shape dimension can be ``-1``.

    broadcast_to : int | tuple[int, ...], optional
        Broadcast the array with :func:`numpy.broadcast_to` to a
        read-only view with the specified shape. Broadcasting is done
        after reshaping (if ``reshape_to`` is not ``None``).

    dtype_out : DTypeLike, optional
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

    to_list : bool, default: False
        Return the validated array as a ``list`` or nested ``list``. Scalar
        values are always returned as a ``Number``  (i.e. ``int`` or ``float``).
        Has no effect if ``to_tuple=True``.

    to_tuple : bool, default: False
        Return the validated array as a ``tuple`` or nested ``tuple``. Scalar
        values are always returned as a ``Number``  (i.e. ``int`` or ``float``).

    get_flags : bool, default: False
        If ``True``, return a ``namedtuple`` of boolean flags with information about
        how the output may differ from the input. The flags returned are:

        - ``same_shape``:  ``True`` if the validated array has the same shape as the input.
          Always ``True`` if ``reshape_to`` and ``broadcast_to`` are ``None``.

        - ``same_dtype``: ``True`` if the validated array has the same dtype as the input.
          Always ``True`` if ``dtype_out`` is ``None``.

        - ``same_type``: ``True`` if the validated array has the same type as the input.
          Always ``True`` if ``return_type`` is ``None`` and the input array
          is supported generically (i.e. is scalar, tuple, list, ndarray).

        - ``same_object``: ``True`` if the validated array is the same object as the input.
          May be ``True`` or ``False`` depending on whether a copy is made.
          See ``copy`` for details.

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
    type_in = type(array) if get_flags else None
    id_in = id(array) if get_flags else None

    array_out = _cast_to_numpy(array, as_any=as_any)

    shape_in = array_out.shape if get_flags else None
    dtype_in = array_out.dtype if get_flags else None

    # Check dtype
    if must_be_real:
        check_real(array_out, name=name)
    if must_have_dtype is not None:
        check_subdtype(array_out, base_dtype=must_have_dtype, name=name)

    # Validate dtype_out
    if (to_list or to_tuple) and np.dtype(dtype_out) not in (
        np.dtype(float),
        np.dtype(int),
        np.dtype(bool),
    ):
        raise ValueError(
            'Invalid `dtype_out` specified. Dtype must be float, int, or bool when \n'
            '`to_list` or `to_tuple` is enabled.',
        )

    # Check shape
    if must_have_shape is not None:
        check_shape(array_out, shape=must_have_shape, name=name)
    if must_have_ndim is not None:
        check_ndim(array_out, ndim=must_have_ndim, name=name)

    # Do reshape _after_ checking shape to prevent unexpected reshaping
    if reshape_to is not None and array_out.shape != reshape_to:
        array_out = array_out.reshape(reshape_to)
    if broadcast_to is not None and array_out.shape != broadcast_to:
        array_out = np.broadcast_to(array_out, broadcast_to, subok=True)

    # Check length _after_ reshaping otherwise length may be wrong
    if (
        must_have_length is not None
        or must_have_min_length is not None
        or must_have_max_length is not None
    ):
        check_length(
            array_out,
            exact_length=must_have_length,
            min_length=must_have_min_length,
            max_length=must_have_max_length,
            allow_scalar=True,
            name=name,
        )

    # Check data values
    if must_be_nonnegative:
        check_nonnegative(array_out, name=name)
    # Check finite before setting dtype since dtype change can fail with inf
    if must_be_finite:
        check_finite(array_out, name=name)
    if must_be_integer:
        check_integer(array_out, strict=False, name=name)
    if must_be_in_range is not None:
        check_range(
            array_out,
            must_be_in_range,
            strict_lower=strict_lower_bound,
            strict_upper=strict_upper_bound,
            name=name,
        )
    if must_be_sorted:
        if isinstance(must_be_sorted, dict):
            check_sorted(array_out, **must_be_sorted, name=name)  # type: ignore[arg-type]
        else:
            check_sorted(array_out, name=name)

    # Set dtype
    if dtype_out is not None:
        array_out = array_out.astype(dtype_out, copy=False)

    def _get_flags(_out: NumpyArray[float]) -> _ValidationFlags:
        return _ValidationFlags(
            same_shape=_out.shape == shape_in,
            same_dtype=_out.dtype == np.dtype(dtype_in),
            same_type=type(_out) is type_in,
            same_object=id(_out) == id_in,
        )

    if to_tuple:
        tuple_out = _cast_to_tuple(array_out)
        return (tuple_out, _get_flags(array_out)) if get_flags else tuple_out
    if to_list:
        list_out = array_out.tolist()
        return (list_out, _get_flags(array_out)) if get_flags else list_out
    return (array_out, _get_flags(array_out)) if get_flags else array_out


def validate_axes(
    *axes: Union[MatrixLike[float], VectorLike[float]],
    normalize: bool = True,
    must_be_orthogonal: bool = True,
    must_have_orientation: Literal['right', 'left'] | None = 'right',
    name: str = 'Axes',
) -> NumpyArray[float]:
    """Validate 3D axes vectors.

    By default, the axes are normalized and checked to ensure they are orthogonal and
    have a right-handed orientation.

    Parameters
    ----------
    *axes : VectorLike[float] | MatrixLike[float]
        Axes to be validated. Axes may be specified as a single argument of a 3x3
        array of row vectors or as separate arguments for each 3-element axis vector.

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
            ['right', 'left'],
            must_contain=must_have_orientation,
            name=f'{name} orientation',
        )

    # Validate number of args
    num_args = len(axes)
    if num_args not in (1, 2, 3):
        raise ValueError(
            'Incorrect number of axes arguments. Number of arguments must be either:\n'
            '\tOne arg (a single array with two or three vectors),'
            '\tTwo args (two vectors), or'
            '\tThree args (three vectors).',
        )

    # Validate axes array
    vector2: Optional[NumpyArray[float]] = None
    if num_args == 1:
        axes_array = validate_array(
            axes[0],
            must_have_shape=[(2, 3), (3, 3)],
            name=name,
            dtype_out=float,
        )
        vector0 = axes_array[0]
        vector1 = axes_array[1]
        if len(axes_array) == 3:
            vector2 = axes_array[2]
    else:
        vector0 = validate_array3(axes[0], name=f'{name} Vector[0]')
        vector1 = validate_array3(axes[1], name=f'{name} Vector[1]')
        if num_args == 3:
            vector2 = validate_array3(axes[2], name=f'{name} Vector[2]')

    if vector2 is None:
        if must_have_orientation is None:
            raise ValueError(
                f'{name} orientation must be specified when only two vectors are given.',
            )
        elif must_have_orientation == 'right':
            vector2 = np.cross(vector0, vector1)
        else:
            vector2 = np.cross(vector1, vector0)
    axes_array = np.vstack((vector0, vector1, vector2))
    check_finite(axes_array, name=name)

    if np.any(np.all(np.isclose(axes_array, np.zeros(3)), axis=1)):
        raise ValueError(f'{name} cannot be zeros.')

    # Normalize axes for dot and cross product calcs
    axes_norm = axes_array / np.linalg.norm(axes_array, axis=1).reshape((3, 1))

    # Check non-parallel
    if np.isclose(np.dot(axes_norm[0], axes_norm[1]), 1) or np.isclose(
        np.dot(axes_norm[0], axes_norm[2]),
        1,
    ):
        raise ValueError(f'{name} cannot be parallel.')

    # Check orthogonality
    cross_0_1 = np.cross(axes_norm[0], axes_norm[1])
    cross_1_2 = np.cross(axes_norm[1], axes_norm[2])

    if must_be_orthogonal and not (
        (np.allclose(cross_0_1, axes_norm[2]) or np.allclose(cross_0_1, -axes_norm[2]))
        and (np.allclose(cross_1_2, axes_norm[0]) or np.allclose(cross_1_2, -axes_norm[0]))
    ):
        raise ValueError(f'{name} are not orthogonal.')

    # Check orientation
    if must_have_orientation:
        dot = np.dot(cross_0_1, axes_norm[2])
        if must_have_orientation == 'right' and dot < 0:
            raise ValueError(f'{name} do not have a right-handed orientation.')
        if must_have_orientation == 'left' and dot > 0:
            raise ValueError(f'{name} do not have a left-handed orientation.')

    return axes_norm if normalize else axes_array


def validate_rotation(
    rotation: RotationLike,
    must_have_handedness: Literal['right', 'left'] | None = None,
    name: str = 'Rotation',
):
    """Validate a rotation as a 3x3 matrix.

    The rotation is valid if its transpose equals its inverse and has a determinant
    of ``1`` (right-handed or "proper" rotation) or ``-1`` (left-handed or "improper"
    rotation). By default, right- and left-handed rotations are allowed.
    Use ``must_have_handedness`` to restrict the handedness.

    Parameters
    ----------
    rotation : RotationLike
        3x3 rotation matrix or a SciPy ``Rotation`` object.

    must_have_handedness : 'right' | 'left' | None, default: None
        Check if the rotation has a specific handedness. If ``right``, the
        determinant must be ``1``. If ``left``, the determinant must be ``-1``.
        By default, either handedness is allowed.

    name : str, default: "Rotation"
        Variable name to use in the error messages if any of the
        validation checks fail.

    Returns
    -------
    np.ndarray
        Validated 3x3 rotation matrix.

    Examples
    --------
    Validate a rotation matrix. The identity matrix is used as a toy example.

    >>> import numpy as np
    >>> from pyvista import _validation
    >>> rotation = np.eye(3)
    >>> _validation.validate_rotation(rotation)
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])

    By default, left-handed rotations (which include reflections) are allowed.

    >>> rotation *= -1  # Add reflections
    >>> _validation.validate_rotation(rotation)
    array([[-1., -0., -0.],
           [-0., -1., -0.],
           [-0., -0., -1.]])

    """
    check_contains(
        ['right', 'left', None], must_contain=must_have_handedness, name='must_have_handedness'
    )
    rotation_matrix = validate_transform3x3(rotation, name=name)
    if not np.allclose(np.linalg.inv(rotation_matrix), rotation_matrix.T):
        raise ValueError(f'{name} is not valid. Its inverse must equal its transpose.')

    if must_have_handedness is not None:
        det = np.linalg.det(rotation_matrix)
        if must_have_handedness == 'right' and not det > 0:
            raise ValueError(
                f'{name} has incorrect handedness. Expected a right-handed rotation, but got a left-handed rotation instead.'
            )
        elif must_have_handedness == 'left' and not det < 0:
            raise ValueError(
                f'{name} has incorrect handedness. Expected a left-handed rotation, but got a right-handed rotation instead.'
            )

    return rotation_matrix


def validate_transform4x4(
    transform: TransformLike, /, *, must_be_finite: bool = True, name: str = 'Transform'
) -> NumpyArray[float]:
    """Validate transform-like input as a 4x4 ndarray.

    Parameters
    ----------
    transform : TransformLike
        Transformation matrix as a 3x3 or 4x4 array, 3x3 or 4x4 vtkMatrix, vtkTransform,
        or a SciPy ``Rotation`` instance. If the input is 3x3, the array is padded using
        a 4x4 identity matrix.

    must_be_finite : bool, default: True
        :func:`Check <pyvista.core._validation.check.check_finite>`
        if all elements of the array are finite, i.e. not ``infinity``
        and not Not a Number (``NaN``).

    name : str, default: "Transform"
        Variable name to use in the error messages if any of the
        validation checks fail.

    Returns
    -------
    np.ndarray
        Validated 4x4 transformation matrix.

    See Also
    --------
    validate_transform3x3
        Similar function for 3x3 transforms.

    validate_array
        Generic array validation function.

    """
    check_string(name, name='Name')
    try:
        arr = np.eye(4)  # initialize
        arr[:3, :3] = validate_transform3x3(transform, must_be_finite=must_be_finite, name=name)
    except (ValueError, TypeError):
        if isinstance(transform, _vtk.vtkMatrix4x4):
            arr = _array_from_vtkmatrix(transform, shape=(4, 4))
        elif isinstance(transform, _vtk.vtkTransform):
            arr = _array_from_vtkmatrix(transform.GetMatrix(), shape=(4, 4))
        else:
            try:
                arr = validate_array(
                    transform,  # type: ignore[arg-type]
                    must_have_shape=[(3, 3), (4, 4)],
                    must_be_finite=must_be_finite,
                    name=name,
                )
            except TypeError:
                raise TypeError(
                    'Input transform must be one of:\n'
                    '\tvtkMatrix4x4\n'
                    '\tvtkMatrix3x3\n'
                    '\tvtkTransform\n'
                    '\t4x4 np.ndarray\n'
                    '\t3x3 np.ndarray\n',
                    '\tscipy.spatial.transform.Rotation\n'
                    f'Got {reprlib.repr(transform)} with type {type(transform)} instead.',
                )
    return arr


def validate_transform3x3(
    transform: TransformLike, /, *, must_be_finite: bool = True, name: str = 'Transform'
) -> NumpyArray[float]:
    """Validate transform-like input as a 3x3 ndarray.

    Parameters
    ----------
    transform : RotationLike
        Transformation matrix as a 3x3 array, vtk matrix, or a SciPy ``Rotation``
        instance.

        .. note::

           Although ``RotationLike`` inputs are accepted, no checks are done
           to verify that the transformation is actually a rotation.
           Therefore, any 3x3 transformation is acceptable.

    must_be_finite : bool, default: True
        :func:`Check <pyvista.core._validation.check.check_finite>`
        if all elements of the array are finite, i.e. not ``infinity``
        and not Not a Number (``NaN``).

    name : str, default: "Transform"
        Variable name to use in the error messages if any of the
        validation checks fail.

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
    check_string(name, name='Name')
    if isinstance(transform, _vtk.vtkMatrix3x3):
        return _array_from_vtkmatrix(transform, shape=(3, 3))
    else:
        try:
            return validate_array(
                transform,  # type: ignore[arg-type]
                must_have_shape=(3, 3),
                must_be_finite=must_be_finite,
                name=name,
            )
        except ValueError:
            pass
        except TypeError:
            try:
                from scipy.spatial.transform import Rotation
            except ModuleNotFoundError:  # pragma: no cover
                pass
            else:
                if isinstance(transform, Rotation):
                    # Get matrix output and try validating again
                    return validate_transform3x3(
                        transform.as_matrix(), must_be_finite=must_be_finite, name=name
                    )

    error_message = (
        f'Input transform must be one of:\n'
        '\tvtkMatrix3x3\n'
        '\t3x3 np.ndarray\n'
        '\tscipy.spatial.transform.Rotation\n'
        f'Got {reprlib.repr(transform)} with type {type(transform)} instead.'
    )
    raise TypeError(error_message)


def _array_from_vtkmatrix(
    matrix: _vtk.vtkMatrix3x3 | _vtk.vtkMatrix4x4,
    shape: tuple[Literal[3], Literal[3]] | tuple[Literal[4], Literal[4]],
) -> NumpyArray[float]:
    """Convert a vtk matrix to an array."""
    array = np.zeros(shape)
    for i, j in product(range(shape[0]), range(shape[1])):
        array[i, j] = matrix.GetElement(i, j)
    return array


class _KwargsValidateNumber(TypedDict):
    reshape: bool
    must_have_dtype: Optional[_NumberUnion]
    must_be_finite: bool
    must_be_real: bool
    must_be_integer: bool
    must_be_nonnegative: bool
    must_be_in_range: Optional[VectorLike[float]]
    strict_lower_bound: bool
    strict_upper_bound: bool
    get_flags: bool
    name: str


# Many type ignores needed to make overloads work here but the typing tests should still pass
@overload
def validate_number(  # type: ignore[misc]  # numpydoc ignore=GL08
    num: Union[NumberType, VectorLike[NumberType]],
    /,
    *,
    reshape: bool = ...,
    dtype_out: None = None,
    **kwargs: Unpack[_KwargsValidateNumber],
) -> NumberType: ...


@overload
def validate_number(  # type: ignore[misc]  # numpydoc ignore=GL08
    num: Union[NumberType, VectorLike[NumberType]],
    /,
    *,
    reshape: bool = ...,
    dtype_out: type[_NumberType],
    **kwargs: Unpack[_KwargsValidateNumber],
) -> _NumberType: ...


def validate_number(  # type: ignore[misc]  # numpydoc ignore=PR01,PR02  # noqa: D417
    num: NumberType | VectorLike[NumberType],
    /,
    *,
    reshape: bool = True,
    must_be_finite: bool = True,
    must_be_real: bool = True,
    must_have_dtype: Optional[_NumberUnion] = None,
    must_be_integer: bool = False,
    must_be_nonnegative: bool = False,
    must_be_in_range: Optional[VectorLike[float]] = None,
    strict_lower_bound: bool = False,
    strict_upper_bound: bool = False,
    dtype_out: Optional[type[_NumberType]] = None,
    get_flags: bool = False,
    name: str = 'Number',
):
    """Validate a real number.

    This function is similar to :func:`~validate_array`, but is configured
    to only allow inputs with one element. The number is checked to be finite
    by default, and the return type is fixed to always return a ``float``,
    ``int``, or ``bool`` type.

    Parameters
    ----------
    num : float | VectorLike[float]
        Number to validate.

    reshape : bool, default: True
        If ``True``, 1D arrays with 1 element are considered valid input
        and are reshaped to be 0-dimensional.

    Other Parameters
    ----------------
    **kwargs
        See :func:`~validate_array` for documentation on all other keyword
        arguments.

    Returns
    -------
    float | int | bool
        Validated number.

    See Also
    --------
    validate_array
        Generic array validation function.

    check_number
        Similar function with fewer options and no return value.

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
    must_have_shape: Union[_ShapeLike, list[_ShapeLike]]
    must_have_shape = [(), (1,)] if reshape else ()

    return validate_array(  # type: ignore[type-var, misc]
        num,
        # Override default vales for these params:
        to_list=True,
        reshape_to=(),
        must_have_shape=must_have_shape,
        # Allow these params to be set by user:
        dtype_out=dtype_out,  # type: ignore[arg-type]
        must_have_dtype=must_have_dtype,
        must_be_nonnegative=must_be_nonnegative,
        must_be_finite=must_be_finite,
        must_be_real=must_be_real,
        must_be_integer=must_be_integer,
        must_be_in_range=must_be_in_range,
        strict_lower_bound=strict_lower_bound,
        strict_upper_bound=strict_upper_bound,
        get_flags=get_flags,
        name=name,
        # These params are irrelevant for this function:
        must_have_ndim=None,
        must_be_sorted=False,
        must_have_length=None,
        must_have_min_length=None,
        must_have_max_length=None,
        broadcast_to=None,
        as_any=False,
        copy=False,
    )


def validate_data_range(  # numpydoc ignore=PR01,PR02  # noqa: D417
    rng: VectorLike[NumberType],
    /,
    *,
    must_have_dtype: Optional[_NumberUnion] = None,
    must_be_nonnegative: bool = False,
    must_be_finite: bool = False,
    must_be_real: bool = True,
    must_be_integer: bool = False,
    must_be_in_range: Optional[VectorLike[float]] = None,
    strict_lower_bound: bool = False,
    strict_upper_bound: bool = False,
    dtype_out: Optional[type[_NumberType]] = None,
    as_any: bool = True,
    copy: bool = False,
    get_flags: bool = False,
    to_tuple: bool = False,
    to_list: bool = False,
    name: str = 'Data Range',
) -> tuple[_NumberType, _NumberType]:
    """Validate a data range.

    This function is similar to :func:`~validate_array`, but is configured
    to only allow inputs with two values and checks that the first value is
    not greater than the second. The return type is also fixed to always
    return a tuple.

    Parameters
    ----------
    rng : VectorLike[float]
        Range to validate in the form ``(lower_bound, upper_bound)``.

    Other Parameters
    ----------------
    **kwargs
        See :func:`~validate_array` for documentation on all other keyword
        arguments.

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
    (-5, 5.0)

    Add additional constraints if needed, e.g. to ensure the output
    only contains floats.

    >>> _validation.validate_data_range([-5, 5.0], dtype_out=float)
    (-5.0, 5.0)

    """
    return cast(
        tuple[_NumberType, _NumberType],
        validate_array(
            rng,
            # Override default vales for these params:
            must_have_shape=2,
            must_be_sorted=True,
            # Allow these params to be set by user:
            dtype_out=dtype_out,
            must_have_dtype=must_have_dtype,
            must_be_nonnegative=must_be_nonnegative,
            must_be_finite=must_be_finite,
            must_be_real=must_be_real,
            must_be_integer=must_be_integer,
            must_be_in_range=must_be_in_range,
            strict_lower_bound=strict_lower_bound,
            strict_upper_bound=strict_upper_bound,
            as_any=as_any,
            copy=copy,
            get_flags=get_flags,
            to_tuple=to_tuple,
            to_list=to_list,
            name=name,
            # These params are irrelevant for this function:
            must_have_length=None,
            must_have_min_length=None,
            must_have_max_length=None,
            reshape_to=None,
            broadcast_to=None,
        ),
    )


class _KwargsValidateArrayNx3(TypedDict, total=False):
    must_have_dtype: Optional[_NumberUnion]
    must_have_length: Optional[Union[int, VectorLike[int]]]
    must_have_min_length: Optional[int]
    must_have_max_length: Optional[int]
    must_be_nonnegative: bool
    must_be_finite: bool
    must_be_real: bool
    must_be_integer: bool
    must_be_sorted: Union[bool, dict[str, Union[bool, int]]]
    must_be_in_range: Optional[VectorLike[float]]
    strict_lower_bound: bool
    strict_upper_bound: bool
    as_any: bool
    copy: bool
    get_flags: bool
    name: str


@overload
def validate_arrayNx3(  # numpydoc ignore=GL08
    array: VectorLike[NumberType],
    /,
    *,
    reshape: Literal[True] = ...,
    dtype_out: None = None,
    **kwargs: Unpack[_KwargsValidateArrayNx3],
) -> NumpyArray[NumberType]: ...


@overload
def validate_arrayNx3(  # numpydoc ignore=GL08
    array: VectorLike[NumberType],
    /,
    *,
    reshape: Literal[True] = ...,
    dtype_out: type[_NumberType],
    **kwargs: Unpack[_KwargsValidateArrayNx3],
) -> NumpyArray[_NumberType]: ...


@overload
def validate_arrayNx3(  # numpydoc ignore=GL08
    array: MatrixLike[NumberType],
    /,
    *,
    reshape: bool = ...,
    dtype_out: None = None,
    **kwargs: Unpack[_KwargsValidateArrayNx3],
) -> NumpyArray[NumberType]: ...


@overload
def validate_arrayNx3(  # numpydoc ignore=GL08
    array: MatrixLike[NumberType],
    /,
    *,
    reshape: bool = ...,
    dtype_out: type[_NumberType],
    **kwargs: Unpack[_KwargsValidateArrayNx3],
) -> NumpyArray[_NumberType]: ...


def validate_arrayNx3(  # numpydoc ignore=PR01,PR02  # noqa: D417
    array: Union[MatrixLike[NumberType], VectorLike[NumberType]],
    /,
    *,
    reshape: bool = True,
    must_have_dtype: Optional[_NumberUnion] = None,
    must_have_length: Optional[Union[int, VectorLike[int]]] = None,
    must_have_min_length: Optional[int] = None,
    must_have_max_length: Optional[int] = None,
    must_be_nonnegative: bool = False,
    must_be_finite: bool = False,
    must_be_real: bool = True,
    must_be_integer: bool = False,
    must_be_sorted: Union[bool, dict[str, Union[bool, int]]] = False,
    must_be_in_range: Optional[VectorLike[float]] = None,
    strict_lower_bound: bool = False,
    strict_upper_bound: bool = False,
    dtype_out: Optional[type[_NumberType]] = None,
    as_any: bool = True,
    copy: bool = False,
    get_flags: bool = False,
    to_list: bool = False,
    to_tuple: bool = False,
    name: str = 'Array',
):
    """Validate a numeric array with N rows and 3 columns.

    This function is similar to :func:`~validate_array`, but is configured
    to only allow inputs with shape ``(N, 3)`` or which can be reshaped to
    ``(N, 3)``. The return type is also fixed to always return a NumPy array.

    Parameters
    ----------
    array : VectorLike[float] | MatrixLike[float]
        1D or 2D array to validate.

    reshape : bool, default: True
        If ``True``, 1D arrays with 3 elements are considered valid
        input and are reshaped to ``(1, 3)`` to ensure the output is
        two-dimensional.

    Other Parameters
    ----------------
    **kwargs
        See :func:`~validate_array` for documentation on all other keyword
        arguments.

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
    array([[1, 2, 3],
           [4, 5, 6]])

    One-dimensional 3-element arrays are automatically reshaped to 2D.

    >>> _validation.validate_arrayNx3([1, 2, 3])
    array([[1, 2, 3]])

    Add additional constraints.

    >>> _validation.validate_arrayNx3(
    ...     ((1, 2, 3), (4, 5, 6)), must_be_in_range=[0, 10]
    ... )
    array([[1, 2, 3],
           [4, 5, 6]])

    """
    must_have_shape: list[_ShapeLike] = [(-1, 3)]
    reshape_to: Optional[_ShapeLike] = None
    if reshape:
        must_have_shape.append((3,))
        reshape_to = (-1, 3)

    return validate_array(  # type: ignore[call-overload, misc]
        array,
        # Override default vales for these params:
        reshape_to=reshape_to,
        must_have_shape=must_have_shape,
        # Allow these params to be set by user:
        must_have_dtype=must_have_dtype,
        must_have_length=must_have_length,
        must_have_min_length=must_have_min_length,
        must_have_max_length=must_have_max_length,
        must_be_nonnegative=must_be_nonnegative,
        must_be_finite=must_be_finite,
        must_be_real=must_be_real,
        must_be_integer=must_be_integer,
        must_be_sorted=must_be_sorted,
        must_be_in_range=must_be_in_range,
        strict_lower_bound=strict_lower_bound,
        strict_upper_bound=strict_upper_bound,
        dtype_out=dtype_out,
        as_any=as_any,
        copy=copy,
        get_flags=get_flags,
        to_list=to_list,
        to_tuple=to_tuple,
        name=name,
        # This parameter is not available
        must_have_ndim=None,
        broadcast_to=None,
    )


class _KwargsValidateArrayN(TypedDict, total=False):
    must_have_dtype: Optional[_NumberUnion]
    must_have_length: Optional[Union[int, VectorLike[int]]]
    must_have_min_length: Optional[int]
    must_have_max_length: Optional[int]
    must_be_nonnegative: bool
    must_be_finite: bool
    must_be_real: bool
    must_be_integer: bool
    must_be_sorted: Union[bool, dict[str, Union[bool, int]]]
    must_be_in_range: Optional[VectorLike[float]]
    strict_lower_bound: bool
    strict_upper_bound: bool
    as_any: bool
    copy: bool
    get_flags: bool
    name: str


@overload
def validate_arrayN(  # numpydoc ignore=GL08
    array: NumberType,
    /,
    *,
    reshape: Literal[True] = ...,
    dtype_out: None = None,
    **kwargs: Unpack[_KwargsValidateArrayN],
) -> NumpyArray[NumberType]: ...


@overload
def validate_arrayN(  # numpydoc ignore=GL08
    array: NumberType,
    /,
    *,
    reshape: Literal[True] = ...,
    dtype_out: type[_NumberType],
    **kwargs: Unpack[_KwargsValidateArrayN],
) -> NumpyArray[_NumberType]: ...


@overload
def validate_arrayN(  # numpydoc ignore=GL08
    array: MatrixLike[NumberType],
    /,
    *,
    reshape: Literal[True] = ...,
    dtype_out: None = None,
    **kwargs: Unpack[_KwargsValidateArrayN],
) -> NumpyArray[NumberType]: ...


@overload
def validate_arrayN(  # numpydoc ignore=GL08
    array: MatrixLike[NumberType],
    /,
    *,
    reshape: Literal[True] = ...,
    dtype_out: type[_NumberType],
    **kwargs: Unpack[_KwargsValidateArrayN],
) -> NumpyArray[_NumberType]: ...


@overload
def validate_arrayN(  # numpydoc ignore=GL08
    array: VectorLike[NumberType],
    /,
    *,
    reshape: bool = ...,
    dtype_out: None = None,
    **kwargs: Unpack[_KwargsValidateArrayN],
) -> NumpyArray[NumberType]: ...


@overload
def validate_arrayN(  # numpydoc ignore=GL08
    array: VectorLike[NumberType],
    /,
    *,
    reshape: bool = ...,
    dtype_out: type[_NumberType],
    **kwargs: Unpack[_KwargsValidateArrayN],
) -> NumpyArray[_NumberType]: ...


def validate_arrayN(  # numpydoc ignore=PR01,PR02  # noqa: D417
    array: Union[NumberType, VectorLike[NumberType], MatrixLike[NumberType]],
    /,
    *,
    reshape: bool = True,
    must_have_dtype: Optional[_NumberUnion] = None,
    must_have_length: Optional[Union[int, VectorLike[int]]] = None,
    must_have_min_length: Optional[int] = None,
    must_have_max_length: Optional[int] = None,
    must_be_nonnegative: bool = False,
    must_be_finite: bool = False,
    must_be_real: bool = True,
    must_be_integer: bool = False,
    must_be_sorted: Union[bool, dict[str, Union[bool, int]]] = False,
    must_be_in_range: Optional[VectorLike[float]] = None,
    strict_lower_bound: bool = False,
    strict_upper_bound: bool = False,
    dtype_out: Optional[type[_NumberType]] = None,
    as_any: bool = True,
    copy: bool = False,
    get_flags: bool = False,
    to_list: bool = False,
    to_tuple: bool = False,
    name: str = 'Array',
):
    """Validate a flat array with N elements.

    This function is similar to :func:`~validate_array`, but is configured
    to only allow inputs with shape ``(N,)`` or which can be reshaped to
    ``(N,)``. The return type is also fixed to always return a NumPy array.

    Parameters
    ----------
    array : float | VectorLike[float] | MatrixLike[float]
        Array-like input to validate.

    reshape : bool, default: True
        If ``True``, 0-dimensional scalars are reshaped to ``(1,)`` and 2D
        vectors with shape ``(1, N)`` are reshaped to ``(N,)`` to ensure the
        output is consistently one-dimensional. Otherwise, all scalar and
        2D inputs are not considered valid.

    Other Parameters
    ----------------
    **kwargs
        See :func:`~validate_array` for documentation on all other keyword
        arguments.

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
    array([1, 2, 3, 4])

    Scalar 0-dimensional values are automatically reshaped to be 1D.

    >>> _validation.validate_arrayN(42.0)
    array([42.0])

    2D arrays where the first dimension is unity are automatically
    reshaped to be 1D.

    >>> _validation.validate_arrayN([[1, 2]])
    array([1, 2])

    Add additional constraints if needed.

    >>> _validation.validate_arrayN((1, 2, 3), must_have_length=3)
    array([1, 2, 3])

    """
    must_have_shape: Union[_ShapeLike, list[_ShapeLike]]
    reshape_to: Optional[tuple[int]] = None
    if reshape:
        must_have_shape = [(), (-1), (1, -1), (-1, 1)]
        reshape_to = (-1,)
    else:
        must_have_shape = (-1,)
    return validate_array(  # type: ignore[call-overload, misc]
        array,
        # Override default vales for these params:
        reshape_to=reshape_to,
        must_have_shape=must_have_shape,
        # Allow these params to be set by user:
        must_have_dtype=must_have_dtype,
        must_have_length=must_have_length,
        must_have_min_length=must_have_min_length,
        must_have_max_length=must_have_max_length,
        must_be_nonnegative=must_be_nonnegative,
        must_be_finite=must_be_finite,
        must_be_real=must_be_real,
        must_be_integer=must_be_integer,
        must_be_sorted=must_be_sorted,
        must_be_in_range=must_be_in_range,
        strict_lower_bound=strict_lower_bound,
        strict_upper_bound=strict_upper_bound,
        dtype_out=dtype_out,
        as_any=as_any,
        copy=copy,
        get_flags=get_flags,
        to_list=to_list,
        to_tuple=to_tuple,
        name=name,
        # This parameter is not available
        must_have_ndim=None,
        broadcast_to=None,
    )


class _KwargsValidateArrayNUnsigned(TypedDict, total=False):
    must_have_dtype: Optional[_NumberUnion]
    must_have_length: Optional[Union[int, VectorLike[int]]]
    must_have_min_length: Optional[int]
    must_have_max_length: Optional[int]
    must_be_real: bool
    must_be_sorted: Union[bool, dict[str, Union[bool, int]]]
    must_be_in_range: Optional[VectorLike[float]]
    strict_lower_bound: bool
    strict_upper_bound: bool
    as_any: bool
    copy: bool
    get_flags: bool
    name: str


_IntegerType = TypeVar('_IntegerType', bound=Union[np.integer, int, np.bool_])  # type: ignore[type-arg]


@overload
def validate_arrayN_unsigned(  # numpydoc ignore=GL08
    array: NumberType,
    /,
    *,
    reshape: Literal[True] = ...,
    dtype_out: int = ...,
    **kwargs: Unpack[_KwargsValidateArrayNUnsigned],
) -> NumpyArray[int]: ...


@overload
def validate_arrayN_unsigned(  # numpydoc ignore=GL08
    array: NumberType,
    /,
    *,
    reshape: Literal[True] = ...,
    dtype_out: type[_IntegerType],
    **kwargs: Unpack[_KwargsValidateArrayNUnsigned],
) -> NumpyArray[_IntegerType]: ...


@overload
def validate_arrayN_unsigned(  # numpydoc ignore=GL08
    array: MatrixLike[NumberType],
    /,
    *,
    reshape: Literal[True] = ...,
    dtype_out: int = ...,
    **kwargs: Unpack[_KwargsValidateArrayNUnsigned],
) -> NumpyArray[int]: ...


@overload
def validate_arrayN_unsigned(  # numpydoc ignore=GL08
    array: MatrixLike[NumberType],
    /,
    *,
    reshape: Literal[True] = ...,
    dtype_out: type[_IntegerType],
    **kwargs: Unpack[_KwargsValidateArrayNUnsigned],
) -> NumpyArray[_IntegerType]: ...


@overload
def validate_arrayN_unsigned(  # numpydoc ignore=GL08
    array: VectorLike[NumberType],
    /,
    *,
    reshape: bool = ...,
    dtype_out: int = ...,
    **kwargs: Unpack[_KwargsValidateArrayNUnsigned],
) -> NumpyArray[int]: ...


@overload
def validate_arrayN_unsigned(  # numpydoc ignore=GL08
    array: VectorLike[NumberType],
    /,
    *,
    reshape: bool = ...,
    dtype_out: type[_IntegerType],
    **kwargs: Unpack[_KwargsValidateArrayNUnsigned],
) -> NumpyArray[_IntegerType]: ...


def validate_arrayN_unsigned(  # type: ignore[misc]  # numpydoc ignore=PR01,PR02  # noqa: D417
    array: Union[NumberType, VectorLike[NumberType], MatrixLike[NumberType]],
    /,
    *,
    reshape: bool = True,
    must_have_dtype: Optional[_NumberUnion] = None,
    must_have_length: Optional[Union[int, VectorLike[int]]] = None,
    must_have_min_length: Optional[int] = None,
    must_have_max_length: Optional[int] = None,
    must_be_real: bool = True,
    must_be_sorted: Union[bool, dict[str, Union[bool, int]]] = False,
    must_be_in_range: Optional[VectorLike[float]] = None,
    strict_lower_bound: bool = False,
    strict_upper_bound: bool = False,
    dtype_out: type[Union[_IntegerType]] = int,  # type: ignore[assignment]
    as_any: bool = True,
    copy: bool = False,
    get_flags: bool = False,
    name: str = 'Array',
):
    """Validate a flat array with N non-negative (unsigned) integers.

    This function is similar to :func:`~validate_array`, but is configured
    to only allow inputs with shape ``(N,)`` or which can be reshaped to
    ``(N,)``. The return type is fixed to always return a NumPy array with
     an integer data type, though an integer subtype may be specified
     (e.g. ``np.unit8``).

    By default, the input is also checked to ensure the values are finite,
    non-negative, and have integer values.

    Parameters
    ----------
    array : float | VectorLike[float] | MatrixLike[float]
        0D, 1D, or 2D array to validate.

    reshape : bool, default: True
        If ``True``, 0-dimensional scalars are reshaped to ``(1,)`` and 2D
        vectors with shape ``(1, N)`` are reshaped to ``(N,)`` to ensure the
        output is consistently one-dimensional. Otherwise, all scalar and
        2D inputs are not considered valid.

    Other Parameters
    ----------------
    **kwargs
        See :func:`~validate_array` for documentation on all other keyword
        arguments.

    Returns
    -------
    np.ndarray
        Validated 1D array with non-negative integers.

    See Also
    --------
    validate_arrayN
        More general function for any numeric one-dimensional array.

    validate_array
        Generic array validation function.

    Examples
    --------
    Validate a 1D array with four non-negative integer-like elements.

    >>> import numpy as np
    >>> from pyvista import _validation
    >>> array = _validation.validate_arrayN_unsigned((1.0, 2.0, 3.0, 4.0))
    >>> array
    array([1, 2, 3, 4])

    Verify that the output data type is integral.

    >>> np.issubdtype(array.dtype, np.integer)
    True

    Scalar 0-dimensional values are automatically reshaped to be 1D.

    >>> _validation.validate_arrayN_unsigned(42)
    array([42])

    2D arrays where the first dimension is unity are automatically
    reshaped to be 1D.

    >>> _validation.validate_arrayN_unsigned([[1, 2]])
    array([1, 2])

    Add additional constraints if needed.

    >>> _validation.validate_arrayN_unsigned((1, 2, 3), must_be_in_range=[1, 3])
    array([1, 2, 3])

    """
    check_subdtype(dtype_out, (np.integer, np.bool_), name='dtype_out')
    return validate_arrayN(  # type: ignore[misc]
        array,  # type: ignore[arg-type]
        reshape=reshape,
        # Override default vales for these params:
        must_be_integer=True,
        must_be_nonnegative=True,
        must_be_finite=True,
        # Allow these params to be set by user:
        must_have_dtype=must_have_dtype,
        must_have_length=must_have_length,
        must_have_min_length=must_have_min_length,
        must_have_max_length=must_have_max_length,
        must_be_real=must_be_real,
        must_be_sorted=must_be_sorted,
        must_be_in_range=must_be_in_range,
        strict_lower_bound=strict_lower_bound,
        strict_upper_bound=strict_upper_bound,
        dtype_out=dtype_out,
        as_any=as_any,
        copy=copy,
        get_flags=get_flags,
        name=name,
    )


class _KwargsValidateArray3(TypedDict, total=False):
    must_have_dtype: Optional[_NumberUnion]
    must_be_nonnegative: bool
    must_be_finite: bool
    must_be_real: bool
    must_be_integer: bool
    must_be_sorted: Union[bool, dict[str, Union[bool, int]]]
    must_be_in_range: Optional[VectorLike[float]]
    strict_lower_bound: bool
    strict_upper_bound: bool
    as_any: bool
    copy: bool
    get_flags: bool
    name: str


@overload
def validate_array3(  # numpydoc ignore=GL08
    array: NumberType,
    /,
    *,
    reshape: bool = ...,
    broadcast: Literal[True],
    dtype_out: None = None,
    **kwargs: Unpack[_KwargsValidateArray3],
) -> NumpyArray[NumberType]: ...


@overload
def validate_array3(  # numpydoc ignore=GL08
    array: NumberType,
    /,
    *,
    reshape: bool = ...,
    broadcast: Literal[True],
    dtype_out: type[_NumberType],
    **kwargs: Unpack[_KwargsValidateArray3],
) -> NumpyArray[_NumberType]: ...


@overload
def validate_array3(  # numpydoc ignore=GL08
    array: MatrixLike[NumberType],
    /,
    *,
    reshape: Literal[True] = ...,
    broadcast: bool = ...,
    dtype_out: None = None,
    **kwargs: Unpack[_KwargsValidateArray3],
) -> NumpyArray[NumberType]: ...


@overload
def validate_array3(  # numpydoc ignore=GL08
    array: MatrixLike[NumberType],
    /,
    *,
    reshape: Literal[True] = ...,
    broadcast: bool = ...,
    dtype_out: type[_NumberType],
    **kwargs: Unpack[_KwargsValidateArray3],
) -> NumpyArray[_NumberType]: ...


@overload
def validate_array3(  # numpydoc ignore=GL08
    array: VectorLike[NumberType],
    /,
    *,
    reshape: bool = ...,
    broadcast: bool = ...,
    dtype_out: None = None,
    **kwargs: Unpack[_KwargsValidateArray3],
) -> NumpyArray[NumberType]: ...


@overload
def validate_array3(  # numpydoc ignore=GL08
    array: VectorLike[NumberType],
    /,
    *,
    reshape: bool = ...,
    broadcast: bool = ...,
    dtype_out: type[_NumberType],
    **kwargs: Unpack[_KwargsValidateArray3],
) -> NumpyArray[_NumberType]: ...


def validate_array3(  # numpydoc ignore=PR01,PR02  # noqa: D417
    array: NumberType | VectorLike[NumberType] | MatrixLike[NumberType],
    /,
    *,
    reshape: bool = True,
    broadcast: bool = False,
    must_be_finite: bool = False,
    must_be_real: bool = True,
    must_have_dtype: Optional[_NumberUnion] = None,
    must_be_integer: bool = False,
    must_be_nonnegative: bool = False,
    must_be_sorted: Union[bool, dict[str, Union[bool, int]]] = False,
    must_be_in_range: Optional[VectorLike[float]] = None,
    strict_lower_bound: bool = False,
    strict_upper_bound: bool = False,
    dtype_out: Optional[type[_NumberType]] = None,
    as_any: bool = True,
    copy: bool = False,
    get_flags: bool = False,
    to_list: bool = False,
    to_tuple: bool = False,
    name: str = 'Array',
):
    """Validate an array with three numbers.

    This function is similar to :func:`~validate_array`, but is configured
    to only allow inputs with shape ``(3,)`` or which can be reshaped to
    ``(3,)``. The return type is also fixed to always return a NumPy array.

    Parameters
    ----------
    array : float | VectorLike[float] | MatrixLike[float]
        Array to validate.

    reshape : bool, default: True
        If ``True``, 2D vectors with shape ``(1, 3)`` or ``(3, 1)`` are
        considered valid input, and are reshaped to ``(3,)`` to ensure
        the output is consistently one-dimensional.

    broadcast : bool, default: False
        If ``True``, scalar values or 1D arrays with a single element
        are considered valid input and the single value is broadcast to
        a length 3 array.

    Other Parameters
    ----------------
    **kwargs
        See :func:`~validate_array` for documentation on all other keyword
        arguments.

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
    array([1, 2, 3])

    2D 3-element arrays are automatically reshaped to be 1D.

    >>> _validation.validate_array3([[1, 2, 3]])
    array([1, 2, 3])

    Scalar 0-dimensional values can be automatically broadcast as
    a 3-element 1D array.

    >>> _validation.validate_array3(42.0, broadcast=True)
    array([42.0, 42.0, 42.0])

    Add additional constraints if needed.

    >>> _validation.validate_array3((1, 2, 3), must_be_nonnegative=True)
    array([1, 2, 3])

    """
    must_have_shape: list[_ShapeLike] = [(3,)]
    reshape_to: Optional[tuple[int]] = None
    if reshape:
        must_have_shape.append((1, 3))
        must_have_shape.append((3, 1))
        reshape_to = (-1,)
    broadcast_to: Optional[tuple[int, ...]] = None
    if broadcast:
        must_have_shape.append(())  # allow 0D scalars
        must_have_shape.append((1,))  # 1D 1-element vectors
        broadcast_to = (3,)

    return validate_array(  # type: ignore[call-overload, misc]
        array,
        # Override default vales for these params:
        reshape_to=reshape_to,
        broadcast_to=broadcast_to,
        must_have_shape=must_have_shape,
        # Allow these params to be set by user:
        dtype_out=dtype_out,
        must_have_dtype=must_have_dtype,
        must_be_nonnegative=must_be_nonnegative,
        must_be_finite=must_be_finite,
        must_be_real=must_be_real,
        must_be_integer=must_be_integer,
        must_be_sorted=must_be_sorted,
        must_be_in_range=must_be_in_range,
        strict_lower_bound=strict_lower_bound,
        strict_upper_bound=strict_upper_bound,
        as_any=as_any,
        copy=copy,
        get_flags=get_flags,
        to_list=to_list,
        to_tuple=to_tuple,
        name=name,
        # These params are irrelevant for this function:
        must_have_ndim=None,
        must_have_length=None,
        must_have_min_length=None,
        must_have_max_length=None,
    )


def _set_default_kwarg_mandatory(kwargs: dict[str, Any], key: str, default: Any) -> None:
    """Set a kwarg and raise ValueError if not set to its default value."""
    val = kwargs.pop(key, default)
    if val != default:
        calling_fname = inspect.stack()[1].function
        msg = (
            f"Parameter '{key}' cannot be set for function `{calling_fname}`.\n"
            f'Its value is automatically set to `{default}`.'
        )
        raise ValueError(msg)
    kwargs[key] = default


def validate_dimensionality(
    dimensionality: Literal[0, 1, 2, 3, '0D', '1D', '2D', '3D'] | VectorLike[int],
    /,
    *,
    reshape: bool = True,
    **kwargs,
) -> int:
    """Validate a dimensionality.

    By default, the dimensionality is checked to ensure it:

    * is scalar or is an array which can be reshaped as a scalar
    * is an integer in the inclusive range ``[0, 3]``
    * or is a valid alias among ``'0D'``, ``'1D'``, ``'2D'``, or ``'3D'``

    Parameters
    ----------
    dimensionality : Literal[0, 1, 2, 3, '0D', '1D', '2D', '3D'] | ArrayLike
        Number to validate.

    reshape : bool, default: True
        If ``True``, 1D arrays with 1 element are considered valid input
        and are reshaped to be 0-dimensional.

    **kwargs : dict, optional
        Additional keyword arguments passed to :func:`~validate_array`.

    Returns
    -------
    int
        Validated dimensionality.

    Examples
    --------
    Validate a dimensionality.

    >>> from pyvista import _validation
    >>> _validation.validate_dimensionality('1D')
    1

    1D arrays are automatically reshaped.

    >>> _validation.validate_dimensionality([3])
    3

    """
    kwargs.setdefault('name', 'Dimensionality')
    kwargs.setdefault('to_list', True)
    kwargs.setdefault('must_be_finite', True)
    kwargs.setdefault('must_be_in_range', [0, 3])

    dimensionality_as_array = np.asarray(dimensionality)
    if np.issubdtype(dimensionality_as_array.dtype, str):
        dimensionality_as_array = np.char.replace(dimensionality_as_array, 'D', '')

    try:
        dimensionality_as_array = dimensionality_as_array.astype(np.int64)
    except ValueError:
        raise ValueError(
            f'`{dimensionality}` is not a valid dimensionality.'
            ' Use one of [0, 1, 2, 3, "0D", "1D", "2D", "3D"].'
        )

    if reshape:
        shape = [(), (1,)]
        _set_default_kwarg_mandatory(kwargs, 'reshape_to', ())
    else:
        shape = ()  # type: ignore[assignment]
    _set_default_kwarg_mandatory(kwargs, 'must_have_shape', shape)

    return validate_array(dimensionality_as_array, **kwargs)


def _validate_color_sequence(
    color: ColorLike | Sequence[ColorLike],
    n_colors: int | None = None,
) -> tuple[Color, ...]:
    """Validate a color sequence.

    If `n_colors` is specified, the output will have `n` colors. For single-color
    inputs, the color is copied and a sequence of `n` identical colors is returned.
    For inputs with multiple colors, the number of colors in the input must
    match `n_colors`.

    If `n_colors` is None, no broadcasting or length-checking is performed.
    """
    from pyvista.plotting.colors import Color

    try:
        # Assume we have one color
        color_list = [Color(color)]  # type: ignore[arg-type]
        n_colors = 1 if n_colors is None else n_colors
        return tuple(color_list * n_colors)
    except ValueError:
        if isinstance(color, (tuple, list)):
            try:
                color_list = [_validate_color_sequence(c, n_colors=1)[0] for c in color]
                if len(color_list) == 1:
                    n_colors = 1 if n_colors is None else n_colors
                    color_list = color_list * n_colors

                # Only return if we have the correct number of colors
                if n_colors is None or len(color_list) == n_colors:
                    return tuple(color_list)
            except ValueError:
                pass
    raise ValueError(
        f'Invalid color(s):\n'
        f'\t{color}\n'
        f'Input must be a single ColorLike color '
        f'or a sequence of {n_colors} ColorLike colors.',
    )
