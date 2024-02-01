"""Functions for processing array-like inputs."""
from __future__ import annotations

import itertools
from typing import (
    Any,
    Generic,
    Iterable,
    List,
    Literal,
    Protocol,
    Sequence,
    Sized,
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
    # the return types correctly for each overload,so we ignore
    # all [overload-overlap] errors and assume the annotations
    # for the overloads are correct
    @overload
    def __new__(  # type: ignore[overload-overlap]
        cls,
        array: NumpyArray[_NumberType],
    ) -> _NumpyArrayWrapper[_NumberType]:
        ...  # pragma: no cover

    @overload
    def __new__(  # type: ignore[overload-overlap]
        cls,
        array: _NumberType,
    ) -> _ScalarWrapper[_NumberType]:
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
    ) -> _NumberSequenceWrapper[_NumberType]:
        ...  # pragma: no cover

    @overload
    def __new__(
        cls,
        array: _NumberSequence2D[_NumberType],
    ) -> _NumberSequenceWrapper[_NumberType]:
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
        # Wrap 1D or 2D sequences as-is
        if isinstance(array, (tuple, list)):
            if _is_number_sequence_1D(array) or _is_number_sequence_2D(array):
                wrapped1 = object.__new__(_NumberSequenceWrapper)
                wrapped1.__setattr__('_array', array)
                return wrapped1
        # Wrap scalars as-is
        elif isinstance(array, (float, int, np.floating, np.integer, np.bool_)):
            wrapped2 = object.__new__(_ScalarWrapper)
            wrapped2.__setattr__('_array', array)
            return wrapped2

        # Everything else gets wrapped as (and possibly converted to) a numpy array
        wrapped3 = object.__new__(_NumpyArrayWrapper)
        wrapped3.__setattr__('_array', np.asanyarray(array))
        return wrapped3

    def __getattr__(self, item):
        try:
            return self.__getattribute__(item)
        except AttributeError:
            return getattr(self.__getattribute__('_array'), item)


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
            return len(self._array), len(cast(Sized, self._array[0]))

    @property
    def ndim(self) -> Union[Literal[1], Literal[2]]:
        if len(self._array) > 0 and isinstance(self._array[0], Sequence):
            return 2
        else:
            return 1

    @property
    def dtype(self) -> Type[_NumberType]:
        iterable: Iterable
        if self.ndim == 1:
            iterable = self._array
        else:
            iterable = itertools.chain.from_iterable(self._array)  # type: ignore[arg-type]

        # create a set with all dtypes
        # exit early if float
        dtypes = set()
        for element in iterable:  # type: ignore[union-attr]
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
