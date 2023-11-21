"""Type aliases for type hints."""
from typing import Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

from pyvista.core._vtk_core import vtkMatrix3x3, vtkMatrix4x4, vtkTransform

NumpyNumArray = npt.NDArray[np.number]
NumpyFltArray = npt.NDArray[np.floating]
NumpyIntArray = npt.NDArray[np.integer]
NumpyBoolArray = npt.NDArray[np.bool_]
NumpyUINT8Array = npt.NDArray[np.uint8]

Number = Union[int, float]
Vector = Union[NumpyNumArray, Sequence[Number]]
IntVector = Union[NumpyIntArray, Sequence[int]]
FloatVector = Union[NumpyFltArray, Sequence[float]]
BoolVector = Union[NumpyBoolArray, Sequence[bool]]
Matrix = Union[NumpyNumArray, Sequence[Vector]]
Array = Union[NumpyNumArray, Sequence[Vector], Sequence[Sequence[Vector]]]
IntArray = Union[NumpyIntArray, Sequence[IntVector], Sequence[Sequence[IntVector]]]
IntMatrix = Union[NumpyIntArray, Sequence[IntVector]]
FloatMatrix = Union[NumpyFltArray, Sequence[FloatVector]]
TransformLike = Union[FloatMatrix, vtkMatrix3x3, vtkMatrix4x4, vtkTransform]
BoundsLike = Tuple[Number, Number, Number, Number, Number, Number]
