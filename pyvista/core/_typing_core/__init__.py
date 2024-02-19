"""Type aliases for type hints."""

# Necessary for autodoc_type_aliases to recognize the type aliases
from __future__ import annotations

from ._aliases import (  # noqa: F401
    Array,
    BoundsLike,
    CellArrayLike,
    CellsLike,
    Matrix,
    Number,
    TransformLike,
    Vector,
)
from ._array_like import NumberType, NumpyArray  # noqa: F401

# NOTE:
# Type aliases are automatically expanded in the documentation.
# To document an alias as-is without expansion, the alias should be added to
# the "autodoc_type_aliases" dictionary in /doc/source/conf.py and added
# to /doc/core/typing.rst
#
# Long or complex type aliases (e.g. a union of 4 or more base types) should
# always be added to the dictionary and documented

NumberType.__doc__ = """Type variable for numeric data types."""
Vector.__doc__ = """One-dimensional array-like object with numerical values.

Includes sequences and numpy arrays."""
Matrix.__doc__ = """Two-dimensional array-like object with numerical values.

Includes singly-nested sequences and numpy arrays."""
Array.__doc__ = """Any-dimensional array-like object with numerical values.

Includes sequences, nested sequences, and numpy arrays up to four dimensions.
Scalar values are not included."""
TransformLike.__doc__ = """Array or vtk object representing a 3x3 or 4x4 matrix."""
BoundsLike.__doc__ = """Tuple of six values representing 3D bounds.

Has the form (xmin, xmax, ymin, ymax, zmin, zmax)."""

# CellsLike.__doc__
# CellArrayLike.__doc__
