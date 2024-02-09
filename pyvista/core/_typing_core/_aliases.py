"""Core type aliases."""

from typing import Tuple, Union

from pyvista.core._vtk_core import vtkCellArray, vtkMatrix3x3, vtkMatrix4x4, vtkTransform

from ._array_like import _ArrayLike, _ArrayLike1D, _ArrayLike2D, _NumberType

Number = Union[int, float]

Vector = _ArrayLike1D[_NumberType]
Matrix = _ArrayLike2D[_NumberType]
Array = _ArrayLike[_NumberType]

TransformLike = Union[Matrix[float], vtkMatrix3x3, vtkMatrix4x4, vtkTransform]
BoundsLike = Tuple[Number, Number, Number, Number, Number, Number]
CellsLike = Union[Matrix[int], Vector[int]]
CellArrayLike = Union[CellsLike, vtkCellArray]
