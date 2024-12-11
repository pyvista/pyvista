"""Type aliases for type hints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._aliases import ArrayLike
from ._aliases import BoundsTuple
from ._aliases import CellArrayLike
from ._aliases import CellsLike
from ._aliases import MatrixLike
from ._aliases import Number
from ._aliases import RotationLike
from ._aliases import TransformLike
from ._aliases import VectorLike
from ._array_like import NumberType
from ._array_like import NumpyArray

if TYPE_CHECKING:  # pragma: no cover
    # Avoid circular imports
    from ._dataset_types import ConcreteGridType
    from ._dataset_types import ConcretePointGridType
    from ._dataset_types import DataObjectType
    from ._dataset_types import DataSetType
    from ._dataset_types import _PointSetType
