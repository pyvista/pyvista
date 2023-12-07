"""Type aliases for type hints."""
from typing import Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

from pyvista.core._vtk_core import vtkMatrix3x3, vtkMatrix4x4, vtkTransform

# To be renamed as: "NumpyArray<type>" to be consistent with generics syntax
# Alternatively, remove the NDArray aliases altogether and use generics syntax
# but with builtin types, e.g. NumpyArray[int] instead of NDArray[np.integer]
NumpyNumArray = npt.NDArray[np.number]
NumpyFltArray = npt.NDArray[np.floating]
NumpyIntArray = npt.NDArray[np.integer]
NumpyBoolArray = npt.NDArray[np.bool_]
NumpyUINT8Array = npt.NDArray[np.uint8]

Number = Union[int, float]

# To be redefined as: Vector = _ArrayLike1D[T]
Vector = Union[NumpyNumArray, Sequence[Number]]

# To be removed: Use Vector[int], Vector[float], Vector[bool] instead
IntVector = Union[NumpyIntArray, Sequence[int]]
FloatVector = Union[NumpyFltArray, Sequence[float]]
BoolVector = Union[NumpyBoolArray, Sequence[bool]]

# To be redefined as: Matrix = _ArrayLike2D[T]
Matrix = Union[NumpyNumArray, Sequence[Vector]]

# To be removed: Use Matrix[int], Matrix[float] instead
IntMatrix = Union[NumpyIntArray, Sequence[IntVector]]
FloatMatrix = Union[NumpyFltArray, Sequence[FloatVector]]

# To be redefined as -> Array = Union[_ArrayLike2D[T], _ArrayLike3D[T], _ArrayLike4D[T]]
Array = Union[NumpyNumArray, Sequence[Vector], Sequence[Sequence[Vector]]]

# To be removed: Use Array[int] instead
IntArray = Union[NumpyIntArray, Sequence[IntVector], Sequence[Sequence[IntVector]]]

TransformLike = Union[FloatMatrix, vtkMatrix3x3, vtkMatrix4x4, vtkTransform]
BoundsLike = Tuple[Number, Number, Number, Number, Number, Number]
