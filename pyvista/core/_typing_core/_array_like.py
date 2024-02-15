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

from typing import TypeVar, Union
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

# Create alias of npt.NDArray bound to numeric types only
_NumType = TypeVar('_NumType', bool, int, float, np.bool_, np.int_, np.float64, np.uint8)
NumpyArray = npt.NDArray[_NumType]

# Define generic nested sequence
_T = TypeVar('_T')
_FiniteNestedSequence = Union[  # Note: scalar types are excluded
    Sequence[_T],
    Sequence[Sequence[_T]],
    Sequence[Sequence[Sequence[_T]]],
    Sequence[Sequence[Sequence[Sequence[_T]]]],
]

_ArrayLike = Union[
    NumpyArray[_NumType],
    _FiniteNestedSequence[_NumType],
    _FiniteNestedSequence[NumpyArray[_NumType]],
]

_ArrayLike1D = Union[
    NumpyArray[_NumType],
    Sequence[_NumType],
    Sequence[NumpyArray[_NumType]],
]
_ArrayLike2D = Union[
    NumpyArray[_NumType],
    Sequence[Sequence[_NumType]],
    Sequence[Sequence[NumpyArray[_NumType]]],
]
_ArrayLike3D = Union[
    NumpyArray[_NumType],
    Sequence[Sequence[Sequence[_NumType]]],
    Sequence[Sequence[Sequence[NumpyArray[_NumType]]]],
]
_ArrayLike4D = Union[
    NumpyArray[_NumType],
    Sequence[Sequence[Sequence[Sequence[_NumType]]]],
    Sequence[Sequence[Sequence[Sequence[NumpyArray[_NumType]]]]],
]
