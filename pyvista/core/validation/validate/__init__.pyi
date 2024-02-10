from typing import Any

import numpy as np

from pyvista.core._typing_core import Vector
from pyvista.core._typing_core._array_like import _ArrayLikeOrScalar, _NumberType

ShapeLike = int | tuple[int, ...] | tuple[()]
DTypeLike = (
    float
    | int
    | bool
    | str
    | np.dtype[np.floating[Any]]
    | np.dtype[np.integer[Any]]
    | np.floating[Any]
    | np.integer[Any]
)

def validate_array(
    array: _ArrayLikeOrScalar[_NumberType],
    *,
    must_have_shape: ShapeLike | None = None,
    must_have_dtype: DTypeLike | None = None,
    must_have_length: int | Vector[int] | None = None,
    must_have_min_length: int | None = None,
    must_have_max_length: int | None = None,
    must_be_nonnegative: bool = False,
    must_be_finite: bool = False,
    must_be_real: bool = True,
    must_be_integer: bool = False,
    must_be_sorted: bool = False,
    must_be_in_range: Vector[float] | None = None,
    strict_lower_bound: bool = False,
    strict_upper_bound: bool = False,
    reshape_to: ShapeLike | None = None,
    broadcast_to: ShapeLike | None = None,
    dtype_out: DTypeLike | None = None,
    as_any: bool = True,
    copy: bool = False,
    output_type: DTypeLike | None = None,
    name: str = 'Array',
): ...

from pyvista.core import _vtk_core as _vtk
from pyvista.core._typing_core import (
    Matrix as Matrix,
    NumpyArray as NumpyArray,
    TransformLike as TransformLike,
)

def validate_axes(
    *axes: Matrix[float] | Vector[float],
    normalize: bool = True,
    must_be_orthogonal: bool = True,
    must_have_orientation: str | None = 'right',
    name: str = 'Axes',
) -> NumpyArray[float]: ...
def validate_transform4x4(transform: TransformLike, *, name: str = 'Transform'): ...
def validate_transform3x3(
    transform: Matrix[float] | _vtk.vtkMatrix3x3, *, name: str = 'Transform'
): ...
def validate_number(
    num: _NumberType | Vector[_NumberType], *, reshape: bool = True, **kwargs
) -> _NumberType: ...
def validate_data_range(rng: Vector[_NumberType], **kwargs): ...
def validate_arrayNx3(
    array: Matrix[_NumberType] | Vector[_NumberType], *, reshape: bool = True, **kwargs
) -> NumpyArray[_NumberType]: ...
def validate_arrayN(
    array: _NumberType | Vector[_NumberType] | Matrix[_NumberType],
    *,
    reshape: bool = True,
    **kwargs,
): ...
def validate_arrayN_uintlike(
    array: _NumberType | Vector[_NumberType] | Matrix[_NumberType],
    *,
    reshape: bool = True,
    **kwargs,
): ...
def validate_array3(
    array: _NumberType | Vector[_NumberType] | Matrix[_NumberType],
    *,
    reshape: bool = True,
    broadcast: bool = False,
    **kwargs,
): ...
