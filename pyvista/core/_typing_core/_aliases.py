"""Core type aliases."""
from typing import Tuple, Union

from pyvista.core._vtk_core import vtkMatrix3x3, vtkMatrix4x4, vtkTransform

from ._array_like import NumberType, _ArrayLike1D, _ArrayLike2D, _ArrayLike3D, _ArrayLike4D

Number = Union[int, float]

# TODO: add --disallow-any-generics option to mypy config to prevent use of
#  type unbound aliases. E.g. the use of `Vector`, which will default to
#  `Vector[Any]` should not be allowed.
Vector = _ArrayLike1D[NumberType]
Matrix = _ArrayLike2D[NumberType]
Array = Union[
    _ArrayLike1D[NumberType],
    _ArrayLike2D[NumberType],
    _ArrayLike3D[NumberType],
    _ArrayLike4D[NumberType],
]

TransformLike = Union[Matrix[float], vtkMatrix3x3, vtkMatrix4x4, vtkTransform]
BoundsLike = Tuple[Number, Number, Number, Number, Number, Number]
