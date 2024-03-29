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

# Create alias of npt.NDArray bound to numeric types only
NumberType = TypeVar('NumberType', bool, int, float, np.bool_, np.int_, np.float64, np.uint8)
NumberType.__doc__ = """Type variable for numeric data types."""
NumpyArray = npt.NDArray[NumberType]

# Define generic nested sequence
_T = TypeVar('_T')
_FiniteNestedSequence = Union[  # Note: scalar types are excluded
    Sequence[_T],
    Sequence[Sequence[_T]],
    Sequence[Sequence[Sequence[_T]]],
    Sequence[Sequence[Sequence[Sequence[_T]]]],
]

_ArrayLike = Union[
    NumpyArray[NumberType],
    _FiniteNestedSequence[NumberType],
    _FiniteNestedSequence[NumpyArray[NumberType]],
]

_ArrayLike1D = Union[
    NumpyArray[NumberType],
    Sequence[NumberType],
    Sequence[NumpyArray[NumberType]],
]
_ArrayLike2D = Union[
    NumpyArray[NumberType],
    Sequence[Sequence[NumberType]],
    Sequence[Sequence[NumpyArray[NumberType]]],
]
_ArrayLike3D = Union[
    NumpyArray[NumberType],
    Sequence[Sequence[Sequence[NumberType]]],
    Sequence[Sequence[Sequence[NumpyArray[NumberType]]]],
]
_ArrayLike4D = Union[
    NumpyArray[NumberType],
    Sequence[Sequence[Sequence[Sequence[NumberType]]]],
    Sequence[Sequence[Sequence[Sequence[NumpyArray[NumberType]]]]],
]
