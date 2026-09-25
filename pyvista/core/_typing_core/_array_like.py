"""Generic array-like type definitions.

The aliases are generic over the Python number type of a sequence's items. NumPy arrays
of any integer, floating, or boolean dtype are accepted whatever the parameter.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeVar

_Scalar = (
    np.float64
    | np.float32
    | np.float16
    | np.int64
    | np.int32
    | np.int16
    | np.int8
    | np.uint64
    | np.uint32
    | np.uint16
    | np.uint8
    | np.bool_
)

_NumberT = TypeVar('_NumberT', bound=float, default=float)

_ScalarT = TypeVar('_ScalarT', bound=np.generic)

# Forwarded as the deprecated `pyvista.NumpyArray`
NumpyArray = npt.NDArray[_ScalarT]

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
    npt.NDArray[_Scalar],
    Sequence[_NumberT],
    Sequence[npt.NDArray[_Scalar]],
]
_ArrayLike2D = Union[
    npt.NDArray[_Scalar],
    Sequence[Sequence[_NumberT]],
    Sequence[Sequence[npt.NDArray[_Scalar]]],
]
_ArrayLike3D = Union[
    npt.NDArray[_Scalar],
    Sequence[Sequence[Sequence[_NumberT]]],
    Sequence[Sequence[Sequence[npt.NDArray[_Scalar]]]],
]
_ArrayLike4D = Union[
    npt.NDArray[_Scalar],
    Sequence[Sequence[Sequence[Sequence[_NumberT]]]],
    Sequence[Sequence[Sequence[Sequence[npt.NDArray[_Scalar]]]]],
]
_ArrayLike = Union[
    _ArrayLike1D[_NumberT],
    _ArrayLike2D[_NumberT],
    _ArrayLike3D[_NumberT],
    _ArrayLike4D[_NumberT],
]
