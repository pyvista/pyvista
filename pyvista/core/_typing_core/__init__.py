"""Type aliases for type hints."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._aliases import (  # noqa: F401
        ArrayLike,
        BoundsLike,
        CellArrayLike,
        CellsLike,
        MatrixLike,
        Number,
        TransformLike,
        VectorLike,
    )
    from ._array_like import NumberType, NumpyArray  # noqa: F401
