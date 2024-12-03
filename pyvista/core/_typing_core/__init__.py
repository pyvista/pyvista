"""Type aliases for type hints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._aliases import ArrayLike  # noqa: F401
from ._aliases import BoundsTuple  # noqa: F401
from ._aliases import CellArrayLike  # noqa: F401
from ._aliases import CellsLike  # noqa: F401
from ._aliases import MatrixLike  # noqa: F401
from ._aliases import Number  # noqa: F401
from ._aliases import RotationLike  # noqa: F401
from ._aliases import TransformLike  # noqa: F401
from ._aliases import VectorLike  # noqa: F401
from ._array_like import NumberType  # noqa: F401
from ._array_like import NumpyArray  # noqa: F401

if TYPE_CHECKING:  # pragma: no cover
    # Avoid circular imports
    from ._dataset_types import ConcreteDataSetType  # noqa: F401
