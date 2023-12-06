"""Core type aliases."""

from typing import Tuple, Union

from pyvista.core._vtk_core import vtkMatrix3x3, vtkMatrix4x4, vtkTransform

from ._generic_number import _DTypeScalar

# flake8: noqa: F401
from ._number_array import (
    _NumberArray1D,
    _NumberArray2D,
    _NumberArray3D,
    _NumberArray4D,
    _NumberNDArray as NumpyArray,
)

Number_ = Union[int, float]

Vector = _NumberArray1D[_DTypeScalar]
Matrix = _NumberArray2D[_DTypeScalar]
DimensionalArray = Union[
    _NumberArray2D[_DTypeScalar], _NumberArray3D[_DTypeScalar], _NumberArray4D[_DTypeScalar]
]

TransformLike = Union[Matrix[float], vtkMatrix3x3, vtkMatrix4x4, vtkTransform]
BoundsLike = Tuple[Number_, Number_, Number_, Number_, Number_, Number_]
