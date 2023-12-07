"""Generic array type definitions.

Definitions here are loosely based on code in numpy._typing._array_like.
Some key differences include:

- Some npt._array_like definitions explicitly support dual-types for
  handling python and numpy scalar data types separately.
  Here, only a single generic type is used for simplicity.

- The npt._array_like definitions use a recursive _NestedSequence protocol.
  Here, finite sequences are used instead.

- The npt._array_like definitions use a generic _SupportsArray protocol.
  Here, we use `ndarray` directly.

- The npt._array_like definitions include scalar types (e.g. float, int).
  Here they are excluded (i.e. scalars are not considered to be arrays).

"""
from typing import Any, Sequence, TypeVar, Union

import numpy as np

from ._dtype import _DType, _DTypeScalar

# Similar definition to numpy.NDArray, but here we use a specialized TypeVar
# and helper class to support using generics with builtin types (e.g. int, float)
NumpyArray = np.ndarray[Any, np.dtype[_DType[_DTypeScalar]]]

_T = TypeVar("_T")
_FiniteNestedSequence = Union[  # Note: scalar types are excluded
    Sequence[_T],
    Sequence[Sequence[_T]],
    Sequence[Sequence[Sequence[_T]]],
    Sequence[Sequence[Sequence[Sequence[_T]]]],
]

_ArrayLike = Union[
    NumpyArray[_DTypeScalar],
    _FiniteNestedSequence[_DTypeScalar],
    _FiniteNestedSequence[NumpyArray[_DTypeScalar]],
]

_ArrayLike1D = Union[
    NumpyArray[_DTypeScalar],
    Sequence[_DTypeScalar],
    Sequence[NumpyArray[_DTypeScalar]],
]
_ArrayLike2D = Union[
    NumpyArray[_DTypeScalar],
    Sequence[Sequence[_DTypeScalar]],
    Sequence[Sequence[NumpyArray[_DTypeScalar]]],
]
_ArrayLike3D = Union[
    NumpyArray[_DTypeScalar],
    Sequence[Sequence[Sequence[_DTypeScalar]]],
    Sequence[Sequence[Sequence[NumpyArray[_DTypeScalar]]]],
]
_ArrayLike4D = Union[
    NumpyArray[_DTypeScalar],
    Sequence[Sequence[Sequence[Sequence[_DTypeScalar]]]],
    Sequence[Sequence[Sequence[Sequence[NumpyArray[_DTypeScalar]]]]],
]
