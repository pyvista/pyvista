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

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar
from typing import Union

import numpy as np
import numpy.typing as npt

# Define numeric types
NumberType = TypeVar(
    'NumberType',
    bound=Union[np.floating, np.integer, np.bool_, float, int, bool],  # type: ignore[type-arg]
)
NumberType.__doc__ = """Type variable for numeric data types."""

# Create a copy of the typevar which can be used for annotating a second variable.
# Its definition should be identical to `NumberType`
_NumberType = TypeVar(  # noqa: PYI018
    '_NumberType',
    bound=Union[np.floating, np.integer, np.bool_, float, int, bool],  # type: ignore[type-arg]
)

NumpyArray = npt.NDArray[NumberType]

_FiniteNestedList = Union[
    list[NumberType],
    list[list[NumberType]],
    list[list[list[NumberType]]],
    list[list[list[list[NumberType]]]],
]
_FiniteNestedTuple = Union[
    tuple[NumberType],
    tuple[tuple[NumberType]],
    tuple[tuple[tuple[NumberType]]],
    tuple[tuple[tuple[tuple[NumberType]]]],
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
_ArrayLike = Union[
    _ArrayLike1D[NumberType],
    _ArrayLike2D[NumberType],
    _ArrayLike3D[NumberType],
    _ArrayLike4D[NumberType],
]
