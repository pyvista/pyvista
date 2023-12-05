"""Core type aliases."""

from typing import Tuple, Union

from pyvista.core._vtk_core import vtkMatrix3x3, vtkMatrix4x4, vtkTransform

from ._generic_number import _NumberType

# flake8: noqa: F401
from ._number_array import (
    _NumberArray1D,
    _NumberArray2D,
    _NumberArray3D,
    _NumberArray4D,
    _NumberNDArray as NumpyArray,
)

Number_ = Union[int, float]

Vector = _NumberArray1D[_NumberType]
Matrix = _NumberArray2D[_NumberType]
DimensionalArray = Union[
    _NumberArray2D[_NumberType], _NumberArray3D[_NumberType], _NumberArray4D[_NumberType]
]

TransformLike = Union[Matrix[float], vtkMatrix3x3, vtkMatrix4x4, vtkTransform]
BoundsLike = Tuple[Number_, Number_, Number_, Number_, Number_, Number_]
