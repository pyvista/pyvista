"""Generic array-like type definitions.

The aliases are generic over the Python number type of a sequence's items. NumPy arrays
of any integer, floating, or boolean dtype are accepted whatever the parameter.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import numpy as np
from numpy.typing import NDArray
from pyvista_validation._typing._array_like import _Scalar
from typing_extensions import TypeVar

_NumberT = TypeVar('_NumberT', bound=float, default=float)

_ScalarT = TypeVar('_ScalarT', bound=np.generic)

# Forwarded as the deprecated `pyvista.NumpyArray`
NumpyArray = NDArray[_ScalarT]

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
