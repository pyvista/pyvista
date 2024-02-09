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

import sys
import typing
from typing import TYPE_CHECKING, Any, List, Sequence, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt

# Create alias of npt.NDArray bound to numeric types only
# TODO: remove # type: ignore once support for 3.8 is dropped
_NumberType = TypeVar(
    '_NumberType',
    bound=Union[np.floating, np.integer, np.bool_, float, int, bool],  # type: ignore[type-arg]
    covariant=True,
)

if not TYPE_CHECKING and sys.version_info < (3, 9, 0):
    # TODO: Remove this conditional block once support for 3.8 is dropped

    # Numpy's type annotations use a customized generic alias type for
    # python < 3.9.0 (defined in numpy.typing._generic_alias._GenericAlias)
    # which makes it incompatible with built-in generic alias types, e.g.
    # Sequence[NDArray[T]]. As a workaround, we define NDArray types using
    # the private typing._GenericAlias type instead
    dtype = typing._GenericAlias(np.dtype, _NumberType)
    NumpyArray = typing._GenericAlias(np.ndarray, (Any, dtype[_NumberType]))
else:
    NumpyArray = npt.NDArray[_NumberType]


# Define generic nested sequence
_T = TypeVar('_T')
_FiniteNestedSequence = Union[  # Note: scalar types are excluded
    Sequence[_T],
    Sequence[Sequence[_T]],
    Sequence[Sequence[Sequence[_T]]],
    Sequence[Sequence[Sequence[Sequence[_T]]]],
]

# Narrow sequence types to tuples and lists only
_Sequence = Union[List[_T], Tuple[_T, ...]]

# Define nested sequences bound to numeric types only
_NumberSequence1D = Union[List[_NumberType], Tuple[_NumberType, ...]]
_NumberSequence2D = Sequence[_NumberSequence1D[_NumberType]]
_NumberSequence3D = Sequence[_NumberSequence2D[_NumberType]]
_NumberSequence4D = Sequence[_NumberSequence3D[_NumberType]]
_NumberSequence = Union[
    _NumberSequence1D[_NumberType],
    _NumberSequence2D[_NumberType],
    _NumberSequence3D[_NumberType],
    _NumberSequence4D[_NumberType],
]

# Define nested sequences of numpy arrays
_NumpyArraySequence1D = Union[List[NumpyArray[_NumberType]], Tuple[NumpyArray[_NumberType], ...]]
_NumpyArraySequence2D = Sequence[_NumpyArraySequence1D[_NumberType]]
_NumpyArraySequence3D = Sequence[_NumpyArraySequence2D[_NumberType]]
_NumpyArraySequence4D = Sequence[_NumpyArraySequence3D[_NumberType]]

_NumpyArraySequence = Union[
    _NumpyArraySequence1D[_NumberType],
    _NumpyArraySequence2D[_NumberType],
    _NumpyArraySequence3D[_NumberType],
    _NumpyArraySequence4D[_NumberType],
]

_ArrayLike1D = Union[
    NumpyArray[_NumberType],
    _NumberSequence1D[_NumberType],
    _NumpyArraySequence1D[_NumberType],
]
_ArrayLike2D = Union[
    NumpyArray[_NumberType],
    _NumberSequence2D[_NumberType],
    _NumpyArraySequence2D[_NumberType],
]
_ArrayLike3D = Union[
    NumpyArray[_NumberType],
    _NumberSequence3D[_NumberType],
    _NumpyArraySequence3D[_NumberType],
]
_ArrayLike4D = Union[
    NumpyArray[_NumberType],
    _NumberSequence4D[_NumberType],
    _NumpyArraySequence4D[_NumberType],
]
_ArrayLike = Union[
    _ArrayLike1D[_NumberType],
    _ArrayLike2D[_NumberType],
    _ArrayLike3D[_NumberType],
    _ArrayLike4D[_NumberType],
]

_ArrayLikeOrScalar = Union[_NumberType, _ArrayLike[_NumberType]]
