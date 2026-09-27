"""NumPy scalar aliases and TypeVars for array annotations."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from pyvista_validation.typing import Floating as _Floating
from pyvista_validation.typing import Integer as _Integer
from pyvista_validation.typing import Real as _Real
from pyvista_validation.typing import Scalar as _Scalar
from typing_extensions import TypeVar

# Any NumPy number or boolean, including widths the concrete aliases leave out
_NumericScalar: TypeAlias = np.floating | np.integer | np.bool_

# Abstract bounds accept any width; the concrete defaults apply when nothing binds
_FloatingT = TypeVar('_FloatingT', bound=np.floating, default=_Floating)
_IntegerT = TypeVar('_IntegerT', bound=np.integer, default=_Integer)
_RealT = TypeVar('_RealT', bound=np.floating | np.integer, default=_Real)
_ScalarT = TypeVar('_ScalarT', bound=_NumericScalar, default=_Scalar)

# Any NumPy scalar, for helpers that pass the dtype through unchanged
_GenericT = TypeVar('_GenericT', bound=np.generic)
