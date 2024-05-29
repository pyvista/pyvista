"""Type aliases for type hints."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._aliases import (
        ArrayLike,
        BoundsLike,
        CellArrayLike,
        CellsLike,
        MatrixLike,
        Number,
        TransformLike,
        VectorLike,
    )
    from ._array_like import NumberType, NumpyArray
else:
    # Aliases are not defined at runtime
    ArrayLike = None
    BoundsLike = None
    CellArrayLike = None
    CellsLike = None
    MatrixLike = None
    Number = None
    NumberType = None
    NumpyArray = None
    TransformLike = None
    VectorLike = None
