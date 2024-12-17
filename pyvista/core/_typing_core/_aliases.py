"""Core type aliases."""

from __future__ import annotations

import contextlib
from typing import NamedTuple
from typing import Union


from ._array_like import NumberType
from ._array_like import _ArrayLike
from ._array_like import _ArrayLike1D
from ._array_like import _ArrayLike2D

with contextlib.suppress(ImportError):
    from scipy.spatial.transform import Rotation

# NOTE:
# Type aliases are automatically expanded in the documentation.
# To document an alias as-is without expansion, the alias should be:
#   (1) added to the "autodoc_type_aliases" dictionary in /doc/source/conf.py
#   (2) added to /doc/core/typing.rst
#   (3) added to the "numpydoc_validation" excludes in pyproject.toml
#
# Long or complex type aliases (e.g. a union of 4 or more base types) should
# always be added to the dictionary and documented
Number = Union[int, float]

VectorLike = _ArrayLike1D[NumberType]
VectorLike.__doc__ = """One-dimensional array-like object with numerical values.

Includes sequences and numpy arrays.
"""

MatrixLike = _ArrayLike2D[NumberType]
MatrixLike.__doc__ = """Two-dimensional array-like object with numerical values.

Includes singly-nested sequences and numpy arrays.
"""

ArrayLike = _ArrayLike[NumberType]
ArrayLike.__doc__ = """Any-dimensional array-like object with numerical values.

Includes sequences, nested sequences, and numpy arrays. Scalar values are not included.
"""

class BoundsTuple(NamedTuple):
    """Tuple of six values representing 3D bounds.

    Has the form ``(x_min, x_max, y_min, y_max, z_min, z_max)``.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


CellsLike = Union[MatrixLike[int], VectorLike[int]]

# Undocumented alias - should be expanded in docs
_ArrayLikeOrScalar = Union[NumberType, ArrayLike[NumberType]]

