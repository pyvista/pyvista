"""Targets of the deprecated type aliases that ``pyvista.typing`` forwards."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

from numpy.typing import NDArray
from pyvista_validation.typing import Scalar as _Scalar
from typing_extensions import TypeVar

from ._array_types import _GenericT

_NumberT = TypeVar('_NumberT', bound=float, default=float)

# Forwarded as the deprecated `pyvista.NumpyArray`
NumpyArray = NDArray[_GenericT]

# Forwarded as the deprecated `pyvista.Number`
Number = Union[int, float]

_ArrayLike1D = Union[
    NDArray[_Scalar],
    Sequence[_NumberT],
    Sequence[NDArray[_Scalar]],
]
_ArrayLike2D = Union[
    NDArray[_Scalar],
    Sequence[Sequence[_NumberT]],
    Sequence[Sequence[NDArray[_Scalar]]],
]
_ArrayLike3D = Union[
    NDArray[_Scalar],
    Sequence[Sequence[Sequence[_NumberT]]],
    Sequence[Sequence[Sequence[NDArray[_Scalar]]]],
]
_ArrayLike4D = Union[
    NDArray[_Scalar],
    Sequence[Sequence[Sequence[Sequence[_NumberT]]]],
    Sequence[Sequence[Sequence[Sequence[NDArray[_Scalar]]]]],
]
_ArrayLike = Union[
    _ArrayLike1D[_NumberT],
    _ArrayLike2D[_NumberT],
    _ArrayLike3D[_NumberT],
    _ArrayLike4D[_NumberT],
]
