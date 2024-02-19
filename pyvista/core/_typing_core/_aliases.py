"""Core type aliases."""

from __future__ import annotations

from typing import Tuple, Union

from pyvista.core import _vtk_core as _vtk

from ._array_like import NumberType, _ArrayLike1D, _ArrayLike2D, _ArrayLike3D, _ArrayLike4D

# NOTE:
# Type aliases are automatically expanded in the documentation.
# To document an alias as-is without expansion, the alias should be added to
# the "autodoc_type_aliases" dictionary in /doc/source/conf.py and added
# to /doc/core/typing.rst
#
# Long or complex type aliases (e.g. a union of 4 or more base types) should
# always be added to the dictionary and documented
Number = Union[int, float]

Vector = _ArrayLike1D[NumberType]
Vector.__doc__ = """One-dimensional array-like object with numerical values.\n\nIncludes sequences and numpy arrays."""

Matrix = _ArrayLike2D[NumberType]
Matrix.__doc__ = """Two-dimensional array-like object with numerical values.\n\nIncludes singly-nested sequences and numpy arrays."""

Array = Union[
    _ArrayLike1D[NumberType],
    _ArrayLike2D[NumberType],
    _ArrayLike3D[NumberType],
    _ArrayLike4D[NumberType],
]
Array.__doc__ = """Any-dimensional array-like object with numerical values.\n\nIncludes sequences, nested sequences, and numpy arrays up to four dimensions. Scalar values are not included."""

TransformLike = Union[Matrix[float], _vtk.vtkMatrix3x3, _vtk.vtkMatrix4x4, _vtk.vtkTransform]
TransformLike.__doc__ = """Array or vtk object representing a 3x3 or 4x4 matrix."""

BoundsLike = Tuple[Number, Number, Number, Number, Number, Number]
BoundsLike.__doc__ = """Tuple of six values representing 3D bounds.\n\nHas the form (xmin, xmax, ymin, ymax, zmin, zmax)."""

CellsLike = Union[Matrix[int], Vector[int]]
# CellsLike.__doc__ = ...

CellArrayLike = Union[CellsLike, _vtk.vtkCellArray]
# CellArrayLike.__doc__ = ...
