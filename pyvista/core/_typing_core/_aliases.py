"""Core type aliases."""

from typing import Tuple, Union

from pyvista.core._vtk_core import vtkCellArray, vtkMatrix3x3, vtkMatrix4x4, vtkTransform

from ._array_like import _ArrayLike1D, _ArrayLike2D, _NumType

Number = Union[int, float]

VectorLike = _ArrayLike1D[_NumType]
MatrixLike = _ArrayLike2D[_NumType]

TransformLike = Union[MatrixLike[float], vtkMatrix3x3, vtkMatrix4x4, vtkTransform]
BoundsLike = Tuple[Number, Number, Number, Number, Number, Number]
CellsLike = Union[MatrixLike[int], VectorLike[int]]
CellArrayLike = Union[CellsLike, vtkCellArray]
