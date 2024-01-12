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

- The npt._array_like TypeVar is bound to np.generic. Here, the
  TypeVar is bound to a subset of numeric types only.

"""
from typing import Sequence, TypeVar, Union

import numpy as np
import numpy.typing as npt


_NumericType = TypeVar('_NumericType', bool, int, float)
_NumberType = TypeVar('_NumberType', int, float)

# Create alias of npt.NDArray bound to numeric types only
_NumDType = TypeVar('_NumDType', bound=Union[np.floating, np.integer], covariant=True)
NumpyArray = npt.NDArray[_NumDType]

# Define generic nested sequence
_T = TypeVar('_T')
_FiniteNestedSequence = Union[  # Note: scalar types are excluded
    Sequence[_T],
    Sequence[Sequence[_T]],
    Sequence[Sequence[Sequence[_T]]],
    Sequence[Sequence[Sequence[Sequence[_T]]]],
]

_ArrayLike = Union[
    NumpyArray[_NumericType],
    _FiniteNestedSequence[_NumericType],
    _FiniteNestedSequence[NumpyArray[_NumericType]],
]

_ArrayLike1D = Union[
    NumpyArray[_NumericType],
    Sequence[_NumericType],
    Sequence[NumpyArray[_NumericType]],
]
_ArrayLike2D = Union[
    NumpyArray[_NumericType],
    Sequence[Sequence[_NumericType]],
    Sequence[Sequence[NumpyArray[_NumericType]]],
]
_ArrayLike3D = Union[
    NumpyArray[_NumericType],
    Sequence[Sequence[Sequence[_NumericType]]],
    Sequence[Sequence[Sequence[NumpyArray[_NumericType]]]],
]
_ArrayLike4D = Union[
    NumpyArray[_NumericType],
    Sequence[Sequence[Sequence[Sequence[_NumericType]]]],
    Sequence[Sequence[Sequence[Sequence[NumpyArray[_NumericType]]]]],
]
