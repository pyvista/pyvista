"""Core type aliases."""

from typing import Tuple, Union

from pyvista.core._vtk_core import vtkMatrix3x3, vtkMatrix4x4, vtkTransform

# flake8: noqa: F401
from ._arrays import _ArrayLike1D, _ArrayLike2D, _ArrayLike3D, _ArrayLike4D
from ._dtype import _DTypeScalar

Number = Union[int, float]

Vector = _ArrayLike1D[_DTypeScalar]
Matrix = _ArrayLike2D[_DTypeScalar]
Array = Union[_ArrayLike2D[_DTypeScalar], _ArrayLike3D[_DTypeScalar], _ArrayLike4D[_DTypeScalar]]

TransformLike = Union[Matrix[float], vtkMatrix3x3, vtkMatrix4x4, vtkTransform]
BoundsLike = Tuple[Number, Number, Number, Number, Number, Number]
