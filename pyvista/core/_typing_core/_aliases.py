"""Core type aliases."""

from __future__ import annotations

from collections.abc import Sequence
import os
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import NamedTuple
from typing import Union

import numpy as np
from numpy.typing import NDArray
from pyvista_validation.typing import ArrayLikeFloat
from pyvista_validation.typing import MatrixLikeFloat
from pyvista_validation.typing import MatrixLikeInt
from pyvista_validation.typing import Scalar as _Scalar
from pyvista_validation.typing import VectorLikeFloat
from pyvista_validation.typing import VectorLikeInt

from pyvista import _vtk

if TYPE_CHECKING:
    import meshio
    import trimesh

    from pyvista import DataObject
    from pyvista import DataSet
    from pyvista import MultiBlock
    from pyvista import PartitionedDataSet

if TYPE_CHECKING or os.environ.get(
    '_PYVISTA_DOCUMENTATION_BULKY_IMPORTS_ALLOWED'
):  # pragma: no cover
    try:
        from scipy.spatial.transform import Rotation
    except ImportError:
        Rotation = None
else:
    Rotation = None

if Rotation is not None:
    RotationLike = Union[MatrixLikeFloat, _vtk.vtkMatrix3x3, Rotation]
else:
    RotationLike = Union[MatrixLikeFloat, _vtk.vtkMatrix3x3]  # type: ignore[misc]
TransformLike = Union[RotationLike, _vtk.vtkMatrix4x4, _vtk.vtkTransform]


class BoundsTuple(NamedTuple):
    """Tuple of six values representing 3D bounds.

    Has the form ``(x_min, x_max, y_min, y_max, z_min, z_max)``.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def __repr__(self) -> str:
        # Split bounds at decimal and compute padding needed to the left of it
        dot = '.'
        strings = [str(float(val)) for val in self]
        has_dot = [dot in s for s in strings]
        split_strings = [s.split(dot) for s in strings]
        pad_left = max(len(parts[0]) for parts in split_strings)

        # Iterate through fields and align values at the decimal
        lines = []
        fields = self._fields
        field_size = max(len(f) for f in fields)
        name = self.__class__.__name__
        whitespace = (len(name) + 1) * ' '
        for i, items in enumerate(zip(fields, split_strings, strict=True)):
            field, parts = items
            if has_dot[i]:
                left, right = parts
                aligned = f'{left:>{pad_left}}{dot}{right}'
            else:
                left = parts[0]
                aligned = f'{left:>{pad_left}}'
            spacing = '' if i == 0 else whitespace
            comma = '' if i == len(fields) - 1 else ','
            lines.append(f'{spacing}{field:<{field_size}} = {aligned}{comma}')

        joined_lines = '\n'.join(lines)
        return f'{name}({joined_lines})'


CellsLike = Union[MatrixLikeInt, VectorLikeInt]

CellArrayLike = Union[CellsLike, _vtk.vtkCellArray]

# Undocumented alias - should be expanded in docs
_ArrayLikeOrScalar = Union[float, _Scalar, ArrayLikeFloat]

# Array of any dtype, or a sequence of anything
_AnyArrayLike = Union[NDArray[Any], Sequence[Any]]

# Array of any integer, floating or boolean dtype
_NumericArray = NDArray[Union[np.floating, np.integer, np.bool_]]

# Array wrapped as a volume, whose values become point scalars
_VolumeArray = NDArray[Union[_Scalar, np.complex64, np.complex128]]

InteractionEventType = Union[Literal['end', 'start', 'always'], _vtk.vtkCommand.EventIds]

LineStyle = Literal['', '-', '--', ':', '-.', '-..']

# Objects that wrap to a DataSet, MultiBlock or PartitionedDataSet
_MeshLike = Union[
    _vtk.vtkDataSet,
    _vtk.vtkMultiBlockDataSet,
    _vtk.vtkPartitionedDataSet,
    'DataSet',
    'MultiBlock',
    'PartitionedDataSet',
    _VolumeArray,
    VectorLikeFloat,
    MatrixLikeFloat,
    'trimesh.Trimesh',
    'meshio.Mesh',
]
WrappableType = Union[_MeshLike, _vtk.vtkDataObject, 'DataObject', _vtk.vtkDataArray, None]
