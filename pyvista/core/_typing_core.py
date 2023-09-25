"""Type aliases for type hints."""
from typing import List, Sequence, Tuple, Union

import numpy as np

from pyvista.core._vtk_core import vtkMatrix3x3, vtkMatrix4x4, vtkTransform

Vector = Union[List[float], Tuple[float, float, float], np.ndarray]
VectorArray = Union[np.ndarray, Sequence[Vector]]
Number = Union[float, int, np.number]
NumericArray = Union[Sequence[Number], np.ndarray]
BoundsLike = Tuple[Number, Number, Number, Number, Number, Number]
TransformLike = Union[np.ndarray, vtkMatrix3x3, vtkMatrix4x4, vtkTransform]
