"""Core type aliases."""

from typing import Tuple, Union

from pyvista.core._vtk_core import vtkCellArray, vtkMatrix3x3, vtkMatrix4x4, vtkTransform

from ._array_like import _ArrayLike1D, _ArrayLike2D, _ArrayLike3D, _ArrayLike4D, _NumType

Number = Union[int, float]

# TODO: add --disallow-any-generics option to mypy config to prevent use of
#  type unbound aliases. E.g. the use of `Vector`, which will default to
#  `Vector[Any]` should not be allowed.
Vector = _ArrayLike1D[_NumType]
Matrix = _ArrayLike2D[_NumType]
Array = Union[
    _ArrayLike1D[_NumType], _ArrayLike2D[_NumType], _ArrayLike3D[_NumType], _ArrayLike4D[_NumType]
]

TransformLike = Union[Matrix[float], vtkMatrix3x3, vtkMatrix4x4, vtkTransform]
BoundsLike = Tuple[Number, Number, Number, Number, Number, Number]
CellsLike = Union[Matrix[int], Vector[int]]
CellArrayLike = Union[CellsLike, vtkCellArray]
