"""Generic array-like type definitions.

Definitions here are loosely based on code in ``numpy._typing._array_like``.
Some key differences include:

- Some npt._array_like definitions explicitly support dual-types for
  handling Python and NumPy scalar data types separately.
  Here, only a single generic type is used for simplicity.

- The npt._array_like definitions use a recursive _NestedSequence protocol.
  Here, finite sequences are used instead.

- The npt._array_like definitions use a generic _SupportsArray protocol.
  Here, we use ``ndarray`` directly.

- The npt._array_like definitions include scalar types (for example, float, int).
  Here they are excluded (that is, scalars are not considered to be arrays).

- The npt._array_like TypeVar is bound to np.generic. Here, the
  TypeVar is bound to a subset of numeric types only.

"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar
from typing import Union

import numpy as np
import numpy.typing as npt

_NumberT = TypeVar(
    '_NumberT',
    bound=np.floating | np.integer | np.bool_ | float | int | bool,
)

# Forwarded as the deprecated `pyvista.NumpyArray`
NumpyArray = npt.NDArray[_NumberT]

_FiniteNestedList = (
    list[_NumberT]
    | list[list[_NumberT]]
    | list[list[list[_NumberT]]]
    | list[list[list[list[_NumberT]]]]
)
_FiniteNestedTuple = (
    tuple[_NumberT]
    | tuple[tuple[_NumberT]]
    | tuple[tuple[tuple[_NumberT]]]
    | tuple[tuple[tuple[tuple[_NumberT]]]]
)

_ArrayLike1D = Union[
    npt.NDArray[_NumberT],
    Sequence[_NumberT],
    Sequence[npt.NDArray[_NumberT]],
]
_ArrayLike2D = Union[
    npt.NDArray[_NumberT],
    Sequence[Sequence[_NumberT]],
    Sequence[Sequence[npt.NDArray[_NumberT]]],
]
_ArrayLike3D = Union[
    npt.NDArray[_NumberT],
    Sequence[Sequence[Sequence[_NumberT]]],
    Sequence[Sequence[Sequence[npt.NDArray[_NumberT]]]],
]
_ArrayLike4D = Union[
    npt.NDArray[_NumberT],
    Sequence[Sequence[Sequence[Sequence[_NumberT]]]],
    Sequence[Sequence[Sequence[Sequence[npt.NDArray[_NumberT]]]]],
]
_ArrayLike = Union[
    _ArrayLike1D[_NumberT],
    _ArrayLike2D[_NumberT],
    _ArrayLike3D[_NumberT],
    _ArrayLike4D[_NumberT],
]
