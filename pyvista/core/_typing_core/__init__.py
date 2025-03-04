"""Type aliases for type hints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._aliases import ArrayLike as ArrayLike
from ._aliases import BoundsTuple as BoundsTuple
from ._aliases import CellArrayLike as CellArrayLike
from ._aliases import CellsLike as CellsLike
from ._aliases import InteractionEventType as InteractionEventType
from ._aliases import MatrixLike as MatrixLike
from ._aliases import Number as Number
from ._aliases import RotationLike as RotationLike
from ._aliases import TransformLike as TransformLike
from ._aliases import VectorLike as VectorLike
from ._array_like import NumberType as NumberType
from ._array_like import NumpyArray as NumpyArray

if TYPE_CHECKING:
    # Avoid circular imports
    from ._dataset_types import ConcreteDataObjectType as ConcreteDataObjectType
    from ._dataset_types import ConcreteDataSetOrMultiBlockType as ConcreteDataSetOrMultiBlockType
    from ._dataset_types import ConcreteDataSetType as ConcreteDataSetType
    from ._dataset_types import ConcreteGridType as ConcreteGridType
    from ._dataset_types import ConcretePointGridType as ConcretePointGridType
    from ._dataset_types import ConcretePointSetType as ConcretePointSetType
