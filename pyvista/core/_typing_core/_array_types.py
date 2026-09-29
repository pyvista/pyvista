"""NumPy scalar and array aliases and TypeVars for array annotations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from pyvista_validation.typing import Array1D
from pyvista_validation.typing import ArrayLikeFloat
from pyvista_validation.typing import Floating
from pyvista_validation.typing import Integer
from pyvista_validation.typing import Real
from pyvista_validation.typing import Scalar
from typing_extensions import TypeVar

# The validation package's scalar aliases, under private names for pyvista
_Floating: TypeAlias = Floating
_Integer: TypeAlias = Integer
_Real: TypeAlias = Real
_Scalar: TypeAlias = Scalar

# Any NumPy number or boolean, including widths the concrete aliases leave out
_NumericScalar: TypeAlias = np.floating | np.integer | np.bool_

# The members of `VectorLikeFloat` and `MatrixLikeFloat` that are sequences of NumPy values
_VectorSequence: TypeAlias = Sequence[_NumericScalar]
_MatrixSequence: TypeAlias = Sequence[Sequence[_NumericScalar] | Array1D[_NumericScalar]]

# Array of any integer, floating or boolean dtype
_NumericArray: TypeAlias = NDArray[_NumericScalar]

# Array wrapped as a volume, whose values become point scalars
_VolumeArray: TypeAlias = NDArray[_Scalar | np.complex64 | np.complex128]

# Array of any dtype, or a sequence of anything
_AnyArrayLike: TypeAlias = NDArray[Any] | Sequence[Any]

# A float array-like or a single number
_ArrayLikeOrScalar: TypeAlias = float | _Scalar | ArrayLikeFloat

# Abstract bounds accept any width; the concrete defaults apply when nothing binds
_FloatingT = TypeVar('_FloatingT', bound=np.floating, default=_Floating)
_IntegerT = TypeVar('_IntegerT', bound=np.integer, default=_Integer)
_RealT = TypeVar('_RealT', bound=np.floating | np.integer, default=_Real)
_ScalarT = TypeVar('_ScalarT', bound=_NumericScalar, default=_Scalar)

# Any NumPy scalar, for helpers that pass the dtype through unchanged
_GenericT = TypeVar('_GenericT', bound=np.generic)
