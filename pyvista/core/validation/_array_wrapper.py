"""Functions for processing array-like inputs."""
from __future__ import annotations

from abc import abstractmethod
import itertools
import reprlib
from typing import (
    Any,
    Callable,
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
from pyvista.core.validation._cast_array import _cast_to_numpy

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
    # array: _ArrayLikeOrScalar[_NumberType]

    # The input array-like types are complex and mypy cannot infer
    # the return types correctly for each overload, so we ignore
    # all [overload-overlap] errors and assume the annotations
    # for the overloads are correct

    @overload
    def __new__(  # type: ignore[overload-overlap]
        cls,
        _array: _NumberType,
    ) -> _NumberWrapper[_NumberType]:
        ...  # pragma: no cover

    @overload
    def __new__(  # type: ignore[overload-overlap]
        cls,
        _array: _NumpyArraySequence[_NumberType],
    ) -> _NumpyArrayWrapper[_NumberType]:
        ...  # pragma: no cover

    @overload
    def __new__(
        cls,
        _array: _NumberSequence1D[_NumberType],
    ) -> _Sequence1DWrapper[_NumberType]:
        ...  # pragma: no cover

    @overload
    def __new__(
        cls,
        _array: _NumberSequence2D[_NumberType],
    ) -> _Sequence2DWrapper[_NumberType]:
        ...  # pragma: no cover

    @overload
    def __new__(
        cls,
        _array: _NumberSequence3D[_NumberType],
    ) -> _NumpyArrayWrapper[_NumberType]:
        ...  # pragma: no cover

    @overload
    def __new__(
        cls,
        _array: _NumberSequence4D[_NumberType],
    ) -> _NumpyArrayWrapper[_NumberType]:
        ...  # pragma: no cover

    @overload
    def __new__(  # type: ignore[overload-overlap]
        cls,
        _array: NumpyArray[_NumberType],
    ) -> _NumpyArrayWrapper[_NumberType]:
        ...  # pragma: no cover

    def __new__(cls, _array: _ArrayLikeOrScalar[_NumberType], description=None):
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
        # Note:
        # __init__ is not used by this class or subclasses so that already-wrapped
        # inputs can be returned as-is without re-initialization.
        # Instead, attributes are initialized here with __setattr__
        try:
            if isinstance(_array, _ArrayLikeWrapper):
                return _array
            elif isinstance(_array, np.ndarray):
                wrapped1 = object.__new__(_NumpyArrayWrapper)
                wrapped1.__setattr__('_array', _array)
                return wrapped1
            elif _is_NumberSequence1D(_array):
                wrapped2 = object.__new__(_Sequence1DWrapper)
                wrapped2.__setattr__('_array', _array)
                wrapped2.__setattr__('_dtype', None)
                return wrapped2
            elif _is_NumberSequence2D(_array):
                wrapped3 = object.__new__(_Sequence2DWrapper)
                wrapped3.__setattr__('_array', _array)
                wrapped3.__setattr__('_dtype', None)
                return wrapped3
            elif _is_Number(_array):
                wrapped4 = object.__new__(_NumberWrapper)
                wrapped4.__setattr__('_array', _array)
                return wrapped4
            # Everything else gets wrapped as (and possibly converted to) a numpy array
            wrapped5 = object.__new__(_NumpyArrayWrapper)
            wrapped5.__setattr__('_array', _cast_to_numpy(_array))
            return wrapped5
        except (ValueError, TypeError):
            raise ValueError(f"The following array is not valid:\n\t{reprlib.repr(_array)}")

    def __getattr__(self, item):
        try:
            return self.__getattribute__(item)
        except AttributeError:
            return getattr(self._array, item)

    def __repr__(self):
        return f'{self.__class__.__name__}({self._array.__repr__()})'

    @abstractmethod
    def all_func(self, func: Callable[[Any, Any], bool], arg):
        ...

    @abstractmethod
    def as_iterable(self) -> Iterable[_NumberType]:
        ...

    @abstractmethod
    def __call__(self):
        ...
        # This method is used for statically mapping the wrapper type
        # to its internal array type: Type[wrapper[T]] -> Type[array[T]].
        # This effectively makes wrapped objects look like array objects
        # so that mypy won't complain that a wrapped object is used where
        # an array is expected.
        # Otherwise, many `type: ignore`s would be needed, or the wrapper
        # class would need to be added to the array-like type alias


class _NumpyArrayWrapper(_ArrayLikeWrapper[_NumberType]):
    _array: NumpyArray[_NumberType]
    dtype: np.dtype[_NumberType]

    def all_func(self, func: Callable[[Any, Any], bool], arg):
        return np.all(func(self._array, arg))

    def __call__(self) -> NumpyArray[_NumberType]:
        return self  # type: ignore[return-value]

    def as_iterable(self) -> Iterable[_NumberType]:
        return self._array.flatten()


class _BuiltinWrapper(_ArrayLikeWrapper[_NumberType]):
    def all_func(self, func: Callable[[Any, Any], bool], arg):
        return all(func(x, arg) for x in self.as_iterable())


class _NumberWrapper(_BuiltinWrapper[_NumberType]):
    _array: _NumberType

    @property
    def shape(self) -> Tuple[()]:
        return ()

    @property
    def ndim(self) -> Literal[0]:
        return 0

    @property
    def size(self) -> Literal[1]:
        return 1

    @property
    def dtype(self) -> Type[_NumberType]:
        return type(self._array)

    def as_iterable(self) -> Iterable[_NumberType]:
        return (self._array,)

    def __call__(self) -> _NumberType:
        return self  # type: ignore[return-value]


class _Sequence1DWrapper(_BuiltinWrapper[_NumberType]):
    _array: _NumberSequence1D[_NumberType]

    @property
    def shape(self) -> Union[Tuple[int]]:
        return (len(self._array),)

    @property
    def ndim(self) -> Literal[1]:
        return 1

    @property
    def size(self) -> int:
        return len(self._array)

    @property
    def dtype(self) -> Type[_NumberType]:
        self._dtype: Type[_NumberType]
        if self._dtype is None:
            self._dtype = _get_dtype_from_iterable(self.as_iterable())
        return self._dtype

    def as_iterable(self) -> Iterable[_NumberType]:
        return self._array

    def __call__(self) -> _NumberSequence1D[_NumberType]:
        return self  # type: ignore[return-value]


class _Sequence2DWrapper(_BuiltinWrapper[_NumberType]):
    _array: _NumberSequence2D[_NumberType]

    @property
    def shape(self) -> Tuple[int, int]:
        return len(self._array), len(self._array[0])

    @property
    def ndim(self) -> Literal[2]:
        return 2

    @property
    def size(self) -> int:
        return len(self._array) * len(self._array[0])

    @property
    def dtype(self) -> Type[_NumberType]:
        self._dtype: Type[_NumberType]
        if self._dtype is None:
            self._dtype = _get_dtype_from_iterable(self.as_iterable())
        return self._dtype

    def as_iterable(self) -> Iterable[_NumberType]:
        return itertools.chain.from_iterable(self._array)

    def __call__(self) -> _NumberSequence2D[_NumberType]:
        return self  # type: ignore[return-value]


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
# reveal_type(_ArrayLikeWrapper([1]).dtype)
# reveal_type(_ArrayLikeWrapper([[1]]))
# reveal_type(_ArrayLikeWrapper([[[1]]]))
