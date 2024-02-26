"""Core type aliases."""

from typing import Tuple, Union

from pyvista.core._vtk_core import vtkCellArray, vtkMatrix3x3, vtkMatrix4x4, vtkTransform

from ._array_like import NumberType, _ArrayLike1D, _ArrayLike2D

Number = Union[int, float]

VectorLike = _ArrayLike1D[NumberType]
MatrixLike = _ArrayLike2D[NumberType]

TransformLike = Union[MatrixLike[float], vtkMatrix3x3, vtkMatrix4x4, vtkTransform]
BoundsLike = Tuple[Number, Number, Number, Number, Number, Number]
CellsLike = Union[MatrixLike[int], VectorLike[int]]
CellArrayLike = Union[CellsLike, vtkCellArray]
