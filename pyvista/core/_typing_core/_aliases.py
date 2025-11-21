"""Core type aliases."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from typing import Literal
from typing import NamedTuple
from typing import Union

from pyvista.core import _vtk_core as _vtk

from ._array_like import NumberType
from ._array_like import _ArrayLike
from ._array_like import _ArrayLike1D
from ._array_like import _ArrayLike2D

if TYPE_CHECKING or os.environ.get(
    'PYVISTA_DOCUMENTATION_BULKY_IMPORTS_ALLOWED'
):  # pragma: no cover
    try:
        from scipy.spatial.transform import Rotation
    except ImportError:
        Rotation = None
else:
    Rotation = None

# NOTE:
# Type aliases are automatically expanded in the documentation.
# To document an alias as-is without expansion, the alias should be:
#   (1) added to the "autodoc_type_aliases" dictionary in /doc/source/conf.py
#   (2) added to /doc/core/typing.rst
#   (3) added to the "numpydoc_validation" excludes in pyproject.toml
#
# Long or complex type aliases (e.g. a union of 4 or more base types) should
# always be added to the dictionary and documented
Number = Union[int, float]

VectorLike = _ArrayLike1D[NumberType]
VectorLike.__doc__ = """One-dimensional array-like object with numerical values.

Includes sequences and numpy arrays.
"""

MatrixLike = _ArrayLike2D[NumberType]
MatrixLike.__doc__ = """Two-dimensional array-like object with numerical values.

Includes singly-nested sequences and numpy arrays.
"""

ArrayLike = _ArrayLike[NumberType]
ArrayLike.__doc__ = """Any-dimensional array-like object with numerical values.

Includes sequences, nested sequences, and numpy arrays. Scalar values are not included.
"""
if Rotation is not None:
    RotationLike = Union[MatrixLike[float], _vtk.vtkMatrix3x3, Rotation]
else:
    RotationLike = Union[MatrixLike[float], _vtk.vtkMatrix3x3]  # type: ignore[misc]
RotationLike.__doc__ = """Array or object representing a spatial rotation.

Includes 3x3 arrays and SciPy Rotation objects.
"""

TransformLike = Union[RotationLike, _vtk.vtkMatrix4x4, _vtk.vtkTransform]
TransformLike.__doc__ = """Array or object representing a spatial transformation.

Includes 3x3 and 4x4 arrays as well as SciPy Rotation objects."""


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


CellsLike = Union[MatrixLike[int], VectorLike[int]]

CellArrayLike = Union[CellsLike, _vtk.vtkCellArray]

# Undocumented alias - should be expanded in docs
_ArrayLikeOrScalar = Union[NumberType, ArrayLike[NumberType]]

InteractionEventType = Union[Literal['end', 'start', 'always'], _vtk.vtkCommand.EventIds]
InteractionEventType.__doc__ = """Interaction event mostly used for widgets.

Includes both strings such as `end`, 'start' and `always` and `_vtk.vtkCommand.EventIds`.
"""
