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

# To be renamed as: Number_ (should be distinct from typing.Number)
Number = Union[int, float]

# To be redefined as: Vector = _NumberArray1D[T]
Vector = Union[NumpyNumArray, Sequence[Number]]

# To be removed: Use Vector[int], Vector[float], Vector[bool] instead
IntVector = Union[NumpyIntArray, Sequence[int]]
FloatVector = Union[NumpyFltArray, Sequence[float]]
BoolVector = Union[NumpyBoolArray, Sequence[bool]]

# To be redefined as: Matrix = GenericArray2D[T]
Matrix = Union[NumpyNumArray, Sequence[Vector]]

# To be removed: Use Matrix[int], Matrix[float] instead
IntMatrix = Union[NumpyIntArray, Sequence[IntVector]]
FloatMatrix = Union[NumpyFltArray, Sequence[FloatVector]]

# To be redefined as -> DimensionalArray = Union[_NumberArray2D[T], _NumberArray3D[T], _NumberArray4D[T]]
Array = Union[NumpyNumArray, Sequence[Vector], Sequence[Sequence[Vector]]]

# To be removed: Use DimensionalArray[int] instead
IntArray = Union[NumpyIntArray, Sequence[IntVector], Sequence[Sequence[IntVector]]]

TransformLike = Union[FloatMatrix, vtkMatrix3x3, vtkMatrix4x4, vtkTransform]
BoundsLike = Tuple[Number, Number, Number, Number, Number, Number]
