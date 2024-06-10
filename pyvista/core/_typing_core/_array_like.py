"""Generic array-like type definitions.

Definitions here are loosely based on code in numpy._typing._array_like.
Some key differences include:

- Some npt._array_like definitions explicitly support dual-types for
  handling python and numpy scalar data types separately.
  Here, only a single generic type is used for simplicity.

- The npt._array_like definitions use a recursive _NestedSequence protocol.
  Here, finite sequences are used instead.

- The npt._array_like definitions use a generic _SupportsArray protocol.
  Here, we use `ndarray` directly.

- The npt._array_like definitions include scalar types (e.g. float, int).
  Here they are excluded (i.e. scalars are not considered to be arrays).

- The npt._array_like TypeVar is bound to np.generic. Here, the
  TypeVar is bound to a subset of numeric types only.

"""

from __future__ import annotations

import sys
import typing
from typing import TYPE_CHECKING
from typing import Any
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

import numpy as np
import numpy.typing as npt

# Define numeric types
# TODO: remove # type: ignore once support for Python3.8 is dropped
_NumberUnion = Union[Type[np.floating], Type[np.integer], Type[np.bool_], Type[float], Type[int], Type[bool]]  # type: ignore[type-arg]
NumberType = TypeVar(
    'NumberType',
    bound=Union[np.floating, np.integer, np.bool_, float, int, bool],  # type: ignore[type-arg]
)
NumberType.__doc__ = """Type variable for numeric data types."""

# Create a copy of the typevar
_NumberType = TypeVar(  # noqa: PYI018
    '_NumberType',
    bound=Union[np.floating, np.integer, np.bool_, float, int, bool],  # type: ignore[type-arg]
)
_PyNumberType = TypeVar('_PyNumberType', float, int, bool)  # noqa: PYI018
_NpNumberType = TypeVar('_NpNumberType', np.float64, np.int_, np.bool_)  # noqa: PYI018


_T = TypeVar('_T')
if not TYPE_CHECKING and sys.version_info < (3, 9, 0):
    # TODO: Remove this conditional block once support for Python3.8 is dropped

    # Numpy's type annotations use a customized generic alias type for
    # python < 3.9.0 (defined in numpy.typing._generic_alias._GenericAlias)
    # which makes it incompatible with built-in generic alias types, e.g.
    # Sequence[NDArray[T]]. As a workaround, we define NDArray types using
    # the private typing._GenericAlias type instead
    np_dtype = typing._GenericAlias(np.dtype, NumberType)
    _np_floating = typing._GenericAlias(np.floating, _T)
    _np_integer = typing._GenericAlias(np.integer, _T)
    np_dtype_floating = typing._GenericAlias(np.dtype, _np_floating[_T])
    np_dtype_integer = typing._GenericAlias(np.dtype, _np_integer[_T])
    NumpyArray = typing._GenericAlias(np.ndarray, (Any, np_dtype[NumberType]))
else:
    np_dtype = np.dtype[NumberType]
    np_dtype_floating = np.dtype[np.floating[Any]]
    np_dtype_integer = np.dtype[np.integer[Any]]
    # Create alias of npt.NDArray bound to numeric types only
    NumpyArray = npt.NDArray[NumberType]

_FiniteNestedList = Union[
    List[NumberType],
    List[List[NumberType]],
    List[List[List[NumberType]]],
    List[List[List[List[NumberType]]]],
]
_FiniteNestedTuple = Union[
    Tuple[NumberType],
    Tuple[Tuple[NumberType]],
    Tuple[Tuple[Tuple[NumberType]]],
    Tuple[Tuple[Tuple[Tuple[NumberType]]]],
]

_ArrayLike1D = Union[
    NumpyArray[NumberType],
    Sequence[NumberType],
    Sequence[NumpyArray[NumberType]],
]
_ArrayLike2D = Union[
    NumpyArray[NumberType],
    Sequence[Sequence[NumberType]],
    Sequence[Sequence[NumpyArray[NumberType]]],
]
_ArrayLike3D = Union[
    NumpyArray[NumberType],
    Sequence[Sequence[Sequence[NumberType]]],
    Sequence[Sequence[Sequence[NumpyArray[NumberType]]]],
]
_ArrayLike4D = Union[
    NumpyArray[NumberType],
    Sequence[Sequence[Sequence[Sequence[NumberType]]]],
    Sequence[Sequence[Sequence[Sequence[NumpyArray[NumberType]]]]],
]
_ArrayLike = Union[
    _ArrayLike1D[NumberType],
    _ArrayLike2D[NumberType],
    _ArrayLike3D[NumberType],
    _ArrayLike4D[NumberType],
]
