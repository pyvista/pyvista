"""Core type aliases."""

from typing import Tuple, Union

from pyvista.core._vtk_core import vtkCellArray, vtkMatrix3x3, vtkMatrix4x4, vtkTransform

from ._array_like import _ArrayLike, _ArrayLike1D, _ArrayLike2D, _NumberType

Number = Union[int, float]

Vector = _ArrayLike1D[_NumberType]
Matrix = _ArrayLike2D[_NumberType]
Array = _ArrayLike[_NumberType]

Vector = _ArrayLike1D[_NumType]
Matrix = _ArrayLike2D[_NumType]
Array = Union[
    _ArrayLike1D[_NumType], _ArrayLike2D[_NumType], _ArrayLike3D[_NumType], _ArrayLike4D[_NumType]

TransformLike = Union[Matrix[float], vtkMatrix3x3, vtkMatrix4x4, vtkTransform]
BoundsLike = Tuple[Number, Number, Number, Number, Number, Number]
CellsLike = Union[Matrix[int], Vector[int]]
CellArrayLike = Union[CellsLike, vtkCellArray]
