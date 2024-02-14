"""Core type aliases."""

# Necessary for autodoc_type_aliases to recognize the type aliases
from __future__ import annotations

from typing import Tuple, Union

from pyvista.core import _vtk_core as _vtk

from ._array_like import NumberType, _ArrayLike1D, _ArrayLike2D, _ArrayLike3D, _ArrayLike4D

Number = Union[int, float]

Vector = _ArrayLike1D[NumberType]
Matrix = _ArrayLike2D[NumberType]
Array = Union[
    _ArrayLike1D[NumberType],
    _ArrayLike2D[NumberType],
    _ArrayLike3D[NumberType],
    _ArrayLike4D[NumberType],
]

TransformLike = Union[Matrix[float], _vtk.vtkMatrix3x3, _vtk.vtkMatrix4x4, _vtk.vtkTransform]
BoundsLike = Tuple[Number, Number, Number, Number, Number, Number]
CellsLike = Union[Matrix[int], Vector[int]]
CellArrayLike = Union[CellsLike, _vtk.vtkCellArray]
