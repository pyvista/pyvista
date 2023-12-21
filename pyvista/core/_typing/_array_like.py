"""Generic array-like type definitions.

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
from typing import Any, Sequence, TypeVar, Union, TypeAlias

import numpy as np

# Create alias of npt.NDArray
_T = TypeVar("_T")
NumpyArray: TypeAlias = np.ndarray[Any, np.dtype[_T]]

_FiniteNestedSequence = Union[  # Note: scalar types are excluded
    Sequence[_T],
    Sequence[Sequence[_T]],
    Sequence[Sequence[Sequence[_T]]],
    Sequence[Sequence[Sequence[Sequence[_T]]]],
]

_ArrayLike = Union[
    NumpyArray[_T],
    _FiniteNestedSequence[_T],
    _FiniteNestedSequence[NumpyArray[_T]],
]

_ArrayLike1D = Union[
    NumpyArray[_T],
    Sequence[_T],
    Sequence[NumpyArray[_T]],
]
_ArrayLike2D = Union[
    NumpyArray[_T],
    Sequence[Sequence[_T]],
    Sequence[Sequence[NumpyArray[_T]]],
]
_ArrayLike3D = Union[
    NumpyArray[_T],
    Sequence[Sequence[Sequence[_T]]],
    Sequence[Sequence[Sequence[NumpyArray[_T]]]],
]
_ArrayLike4D = Union[
    NumpyArray[_T],
    Sequence[Sequence[Sequence[Sequence[_T]]]],
    Sequence[Sequence[Sequence[Sequence[NumpyArray[_T]]]]],
]
