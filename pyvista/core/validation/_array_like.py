"""Functions for processing array-like inputs."""
from __future__ import annotations

import itertools
from typing import (
    Generic,
    Iterable,
    List,
    Literal,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
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
    _NumpyArraySequence1D,
    _NumpyArraySequence2D,
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
_NumberType_co = TypeVar('_NumberType_co', bound=Union[float, int, np.number], covariant=True)
DTypeLike = Union[np.dtype[_NumberType_co], Type[_NumberType_co], str]


class _ArrayLikeWrapper(Generic[_NumberType]):
    array: _ArrayLikeOrScalar[_NumberType]

    @overload
    def __new__(
        cls,
        array: NumpyArray[_NumberType],
    ) -> _NumpyArrayWrapper[_NumberType]:
        ...

    @overload
    def __new__(
        cls,
        array: _NumberType,
    ) -> _ScalarWrapper[_NumberType]:
        ...

    @overload
    def __new__(
        cls,
        array: _NumpyArraySequence[_NumberType],
    ) -> _NumpyArrayWrapper[_NumberType]:
        ...

    @overload
    def __new__(
        cls,
        array: _NumberSequence1D[_NumberType],
    ) -> _NumberSequenceWrapper[_NumberType]:
        ...

    @overload
    def __new__(
        cls,
        array: _NumberSequence2D[_NumberType],
    ) -> _NumberSequenceWrapper[_NumberType]:
        ...

    @overload
    def __new__(
        cls,
        array: _NumberSequence3D[_NumberType],
    ) -> _NumpyArrayWrapper[_NumberType]:
        ...

    @overload
    def __new__(
        cls,
        array: _NumberSequence4D[_NumberType],
    ) -> _NumpyArrayWrapper[_NumberType]:
        ...

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
        # Wrap 1D or 2D sequences as-as
        if isinstance(array, (tuple, list)):
            array_out: Union[
                _NumberSequence1D[_NumberType],
                _NumberSequence2D[_NumberType],
                _NumpyArraySequence1D[_NumberType],
                _NumpyArraySequence2D[_NumberType],
            ]
            array_out = array
            if _is_number_sequence_1D(array) or _is_number_sequence_2D(array):
                return object.__new__(_NumberSequenceWrapper)
        # Wrap scalars as-is
        elif isinstance(array, (np.floating, np.integer, np.bool_, float, int, bool)):
            # reveal_type(array)
            return object.__new__(_ScalarWrapper)
        # Everything else gets wrapped as (and possibly converted to) a numpy array
        # reveal_type(array)
        return object.__new__(_NumpyArrayWrapper)

    def __init__(self, array):
        self._array = array

    def __getattr__(self, item):
        try:
            return self.__getattribute__(item)
        except AttributeError:
            return getattr(self.__getattribute__('_array'), item)


from typing import Any, Protocol, TypeVar, runtime_checkable

_DType_co = TypeVar('_DType_co', bound=np.generic, covariant=True)


@runtime_checkable
class _SupportsArray(Protocol[_DType_co]):
    def __array__(self) -> np.ndarray[Any, np.dtype[_DType_co]]:
        ...


class _NumpyArrayWrapper(_ArrayLikeWrapper[_NumberType]):
    _array: NumpyArray[_NumberType]
    dtype: np.dtype[_NumberType]

    def __init__(self, array):
        self._array = np.asanyarray(array)


class _ScalarWrapper(_ArrayLikeWrapper[_NumberType]):
    _array: _NumberType

    @property
    def shape(self) -> Tuple[()]:
        return ()

    @property
    def ndim(self) -> Literal[0]:
        return 0

    @property
    def dtype(self) -> Type[_NumberType]:
        return type(self._array)


class _NumberSequenceWrapper(_ArrayLikeWrapper[_NumberType]):
    _array: Union[_NumberSequence1D[_NumberType], _NumberSequence2D[_NumberType]]

    def __init__(self, array):
        super().__init__(array)
        if self.ndim == 2:
            # check all subarray shapes are equal
            sub_shape = len(self._array[0])
            all_same = all(len(sub_array) == sub_shape for sub_array in array)
            if not all_same:
                raise ValueError(
                    "The nested sequence array has an inhomogeneous shape. "
                    "All sub-arrays must have the same shape."
                )

    @property
    def shape(self) -> Union[Tuple[int], Tuple[int, int]]:
        if self.ndim == 1:
            return (len(self._array),)
        else:
            return (len(self._array), len(self._array[0]))

    @property
    def ndim(self) -> Union[Literal[1], Literal[2]]:
        if len(self._array) > 0 and isinstance(self._array[0], Sequence):
            return 2
        else:
            return 1

    @property
    def dtype(self) -> Type[_NumberType]:
        if self.ndim == 1:
            iterable = self._array
        else:
            iterable = itertools.chain.from_iterable(self._array)

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

    # class _NumpyArraySequenceWrapper(_ArrayLikeWrapper[_NumberType]):


#     array: _NumpyArraySequence[_NumberType]

# reveal_type(_ArrayLikeWrapper(1))
# reveal_type(_ArrayLikeWrapper(np.array([1], dtype=int)))
# reveal_type(_ArrayLikeWrapper(np.array([1], dtype=int)).array)
# reveal_type(_ArrayLikeWrapper(1).array)
# reveal_type(_ArrayLikeWrapper(1).dtype)
# reveal_type(_ArrayLikeWrapper([1]).array)
# reveal_type(_ArrayLikeWrapper([[1]]))

_SequenceArgType = TypeVar('_SequenceArgType', Sequence[Sequence], Sequence[np.ndarray])


def _has_element_types(array: Iterable, types: Tuple[Type, ...], N=None):
    """Check that iterable elements have the specified type.

    Parameters
    ----------
    N : int
        Only check the first `N` elements. Can be used to reduce the
        performance cost of this check. Set to `None` to check all elements.
    """
    iterator = itertools.islice(array, N)
    return all(isinstance(item, types) for item in iterator)


def _is_number_sequence_1D(array: Union[Tuple, List], N=None):
    return isinstance(array, (tuple, list)) and _has_element_types(array, (float, int), N=N)


def _is_number_sequence_2D(array: Union[Tuple, List], N=None):
    return (
        isinstance(array, (tuple, list))
        and len(array) > 0
        and all(_is_number_sequence_1D(subarray, N=N) for subarray in array)
    )
