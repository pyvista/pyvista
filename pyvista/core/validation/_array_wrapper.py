"""Functions for processing array-like inputs."""
from __future__ import annotations

import itertools
from typing import (
    Any,
    Generic,
    Iterable,
    Literal,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
    runtime_checkable,
)

import numpy as np

from pyvista.core._typing_core import NumpyArray
from pyvista.core._typing_core._array_like import (
    _ArrayLikeOrScalar,
    _NumberSequence1D,
    _NumberSequence2D,
    _NumberSequence3D,
    _NumberSequence4D,
    _NumberType,
    _NumpyArraySequence,
)
from pyvista.core._typing_core._type_guards import (
    _is_Number,
    _is_NumberSequence1D,
    _is_NumberSequence2D,
)

# Similar definitions to numpy._typing._shape but with modifications:
#  - explicit support for empty tuples `()`
#  - strictly uses tuples for indexing
#  - our ShapeLike definition includes single integers (numpy's does not)

ScalarShape = Tuple[()]
ArrayShape = Tuple[int, ...]
Shape = Union[ScalarShape, ArrayShape]
ShapeLike = Union[int, Shape]

# Similar to npt.DTypeLike but is bound to numeric types
# and does not allow _SupportsDType protocol
DTypeLike = Union[np.dtype, Type[Any], str]


# Define array protocol
# This should match numpy's definition exactly
_DType_co = TypeVar('_DType_co', bound=np.generic, covariant=True)


@runtime_checkable
class _SupportsArray(Protocol[_DType_co]):
    def __array__(self) -> np.ndarray[Any, np.dtype[_DType_co]]:
        ...  # pragma: no cover


class _ArrayLikeWrapper(Generic[_NumberType]):
    array: _ArrayLikeOrScalar[_NumberType]

    # The input array-like types are complex and mypy cannot infer
    # the return types correctly for each overload, so we ignore
    # all [overload-overlap] errors and assume the annotations
    # for the overloads are correct

    @overload
    def __new__(  # type: ignore[overload-overlap]
        cls,
        array: _NumberType,
    ) -> _NumberWrapper[_NumberType]:
        ...  # pragma: no cover

    @overload
    def __new__(  # type: ignore[overload-overlap]
        cls,
        array: _NumpyArraySequence[_NumberType],
    ) -> _NumpyArrayWrapper[_NumberType]:
        ...  # pragma: no cover

    @overload
    def __new__(
        cls,
        array: _NumberSequence1D[_NumberType],
    ) -> _Sequence1DWrapper[_NumberType]:
        ...  # pragma: no cover

    @overload
    def __new__(
        cls,
        array: _NumberSequence2D[_NumberType],
    ) -> _Sequence2DWrapper[_NumberType]:
        ...  # pragma: no cover

    @overload
    def __new__(
        cls,
        array: _NumberSequence3D[_NumberType],
    ) -> _NumpyArrayWrapper[_NumberType]:
        ...  # pragma: no cover

    @overload
    def __new__(
        cls,
        array: _NumberSequence4D[_NumberType],
    ) -> _NumpyArrayWrapper[_NumberType]:
        ...  # pragma: no cover

    @overload
    def __new__(  # type: ignore[overload-overlap]
        cls,
        array: NumpyArray[_NumberType],
    ) -> _NumpyArrayWrapper[_NumberType]:
        ...  # pragma: no cover

    def __new__(
        cls,
        array: _ArrayLikeOrScalar[_NumberType],
    ):
        """Wrap array-like inputs to standardize the representation.

        The following inputs are wrapped as-is without modification:
            - scalar dtypes (e.g. float, int)
            - 1D numeric sequences
            - 2D numeric nested sequences

        The above types are also given `shape`, `dtype`, and `ndim`
        attributes.

        All other array-like inputs (e.g. nested numeric sequences with
        depth > 2, nested sequences of numpy arrays) are cast as a numpy
        array.

        """
        # Do the most common checks first to
        # avoid making unnecessary checks
        if isinstance(array, np.ndarray):
            return object.__new__(_NumpyArrayWrapper)
        if _is_NumberSequence1D(array):
            return object.__new__(_Sequence1DWrapper)
        elif _is_NumberSequence2D(array):
            return object.__new__(_Sequence2DWrapper)
        elif _is_Number(array):
            return object.__new__(_NumberWrapper)

        # Everything else gets wrapped as (and possibly converted to) a numpy array
        return object.__new__(_NumpyArrayWrapper)

    def __getattr__(self, item):
        try:
            return self.__getattribute__(item)
        except AttributeError:
            return getattr(self._array, item)


class _NumpyArrayWrapper(_ArrayLikeWrapper[_NumberType]):
    _array: NumpyArray[_NumberType]
    dtype: np.dtype[_NumberType]

    def __init__(self, array: NumpyArray[_NumberType]):
        self._array = np.asanyarray(array)


class _NumberWrapper(_ArrayLikeWrapper[_NumberType]):
    _array: _NumberType

    def __init__(self, array):
        self._array = array

    @property
    def shape(self) -> Tuple[()]:
        return ()

    @property
    def ndim(self) -> Literal[0]:
        return 0

    @property
    def dtype(self) -> Type[_NumberType]:
        return type(self._array)

    @property
    def iterable(self) -> Iterable[_NumberType]:
        return (self._array,)


class _Sequence1DWrapper(_ArrayLikeWrapper[_NumberType]):
    _array: _NumberSequence1D[_NumberType]

    def __init__(self, array):
        self._array = array
        self._dtype = None

    @property
    def shape(self) -> Union[Tuple[int]]:
        return (len(self._array),)

    @property
    def ndim(self) -> Literal[1]:
        return 1

    @property
    def dtype(self) -> Type[_NumberType]:
        if self._dtype is None:
            self._dtype = _get_dtype_from_iterable(self._array)
        return self._dtype

    @property
    def iterable(self) -> Iterable[_NumberType]:
        return self._array


class _Sequence2DWrapper(_ArrayLikeWrapper[_NumberType]):
    _array: _NumberSequence2D[_NumberType]

    def __init__(self, array):
        self._array = array
        self._dtype = None

    @property
    def shape(self) -> Tuple[int, int]:
        return len(self._array), len(self._array[0])

    @property
    def ndim(self) -> Literal[2]:
        return 2

    @property
    def dtype(self) -> Type[_NumberType]:
        if self._dtype is None:
            self._dtype = _get_dtype_from_iterable(self.iterable)
        return self._dtype

    @property
    def iterable(self) -> Iterable[_NumberType]:
        return itertools.chain.from_iterable(self._array)


def _get_dtype_from_iterable(iterable: Iterable[_NumberType]):
    # Note: This function assumes all elements are numeric.

    # create a set with all dtypes
    # exit early if float
    dtypes = set()
    for element in iterable:
        dtype = type(element)
        if dtype is float:
            return cast(Type[_NumberType], float)
        else:
            dtypes.add(dtype)
    if len(dtypes) == 0:
        return cast(Type[_NumberType], float)
    elif dtypes in [{int}, {bool, int}]:
        return cast(Type[_NumberType], int)
    elif dtypes == {bool}:
        return cast(Type[_NumberType], bool)
    else:
        raise TypeError(f"Unexpected error: dtype should be numeric, got {dtypes} instead.")


# reveal_type(_ArrayLikeWrapper(1))
# reveal_type(_ArrayLikeWrapper(np.array([1], dtype=int)))
# reveal_type(_ArrayLikeWrapper(np.array([1], dtype=int))._array)
# reveal_type(_ArrayLikeWrapper(1)._array)
# reveal_type(_ArrayLikeWrapper(1).dtype)
# reveal_type(_ArrayLikeWrapper([1])._array)
# reveal_type(_ArrayLikeWrapper([[1]]))
# reveal_type(_ArrayLikeWrapper([[[1]]]))
