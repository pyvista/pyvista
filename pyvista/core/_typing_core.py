"""Type aliases for type hints."""
# flake8: noqa: F401
from typing import Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

from pyvista.core._vtk_core import vtkMatrix3x3, vtkMatrix4x4, vtkTransform

from ._typing import Array, BoundsLike, Matrix, Number, NumpyArray, TransformLike, Vector
