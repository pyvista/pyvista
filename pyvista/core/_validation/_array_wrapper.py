"""Functions for processing array-like inputs."""

from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy
import itertools
import reprlib
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterable,
    List,
    Literal,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    overload,
)

import numpy as np

from pyvista.core._typing_core._array_like import NumberType, _FiniteNestedList, _FiniteNestedTuple
from pyvista.core._typing_core._type_guards import (
    _is_NestedNumberSequence,
    _is_Number,
    _is_NumberSequence,
)
from pyvista.core._validation._cast_array import _cast_to_numpy, _cast_to_tuple

if TYPE_CHECKING:  # pragma: no cover
    from pyvista.core._typing_core import NumpyArray
    from pyvista.core._typing_core._aliases import _ArrayLikeOrScalar


class _ArrayLikeWrapper(Generic[NumberType]):

    # The input array-like types are complex and mypy cannot infer
    # the return types correctly for each overload, so we ignore
    # all [overload-overlap] errors and assume the annotations
    # for the overloads are correct

    @overload
    def __new__(cls, _array: NumberType) -> _NumberWrapper[NumberType]: ...  # type: ignore[overload-overlap]

    @overload
    def __new__(cls, _array: List[List[NumberType]]) -> _NestedSequenceWrapper[NumberType]: ...  # type: ignore[overload-overlap]

    @overload
    def __new__(cls, _array: List[NumberType]) -> _SequenceWrapper[NumberType]: ...  # type: ignore[overload-overlap]

    @overload
    def __new__(cls, _array: Tuple[Tuple[NumberType, ...]]) -> _NestedSequenceWrapper[NumberType]: ...  # type: ignore[overload-overlap]

    @overload
    def __new__(cls, _array: Tuple[NumberType, ...]) -> _SequenceWrapper[NumberType]: ...  # type: ignore[overload-overlap]

    @overload
    def __new__(cls, _array: NumpyArray[NumberType]) -> _NumpyArrayWrapper[NumberType]: ...

    @overload
    def __new__(cls, _array: _ArrayLikeOrScalar[NumberType]) -> _NumpyArrayWrapper[NumberType]: ...

    def __new__(cls, _array: _ArrayLikeOrScalar[NumberType]):
        """Wrap array-like inputs to standardize the representation.

        The following inputs are wrapped as-is without modification:
            - scalar dtypes (e.g. float, int)
            - flat numeric sequences
            - nested numeric sequences

        The above types are also given `shape`, `dtype`, `ndim`,
        `size`, and other generic array attributes.

        All other array-like inputs (e.g. nested numeric sequences with
        depth > 2, nested sequences of numpy arrays) are cast as a numpy
        array.

        """
        # Note:
        # __init__ is not used by this class or subclasses so that already-wrapped
        # inputs can be returned as-is without re-initialization.
        # Instead, attributes are initialized here with __setattr__

        # WARNING: Wrapped arrays of built-in types are assumed to be static and
        # should not be mutated since their properties are not dynamically
        # generated. Therefore, do not modify the underlying `_array`.
        try:
            if isinstance(_array, _ArrayLikeWrapper):
                return _array
            elif _is_NumberSequence(_array):
                wrapped2 = object.__new__(_SequenceWrapper)
                wrapped2.__setattr__('_array', _array)
                wrapped2.__setattr__('_dtype', None)
                return wrapped2
            elif _is_NestedNumberSequence(_array):
                wrapped3 = object.__new__(_NestedSequenceWrapper)
                wrapped3.__setattr__('_array', _array)
                wrapped3.__setattr__('_dtype', None)
                return wrapped3
            elif _is_Number(_array):
                wrapped4 = object.__new__(_NumberWrapper)
                wrapped4.__setattr__('_array', _array)
                return wrapped4
            else:
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
    def to_list(self, input_array: _ArrayLikeOrScalar[NumberType], copy: bool): ...

    @abstractmethod
    def to_tuple(self, input_array: _ArrayLikeOrScalar[NumberType], copy: bool): ...

    @abstractmethod
    def to_numpy(self, input_array: _ArrayLikeOrScalar[NumberType], copy: bool): ...

    @abstractmethod
    def all_func(self, func: Callable[..., Any], *args): ...

    @abstractmethod
    def as_iterable(self) -> Iterable[NumberType]: ...

    @abstractmethod
    def change_dtype(self, dtype): ...

    @abstractmethod
    def __call__(self):
        """Return self if called.

        This method is used for statically mapping the wrapper type
        to its internal array type: wrapper[T] -> array[T].
        This effectively makes wrapped objects look like array objects
        so that mypy won't complain that a wrapped object is used where
        an array is expected.
        Otherwise, many `# type: ignore` comments would be needed, or the
        private wrapper class would need to be added to the public array-like
        type alias definition.

        E.g. Pass a wrapped array where an array-like object is expected:

        >>> import numpy as np
        >>> from pyvista.core._validation._array_wrapper import (
        ...     _ArrayLikeWrapper,
        ... )
        >>> from pyvista import _validation
        >>> wrapped = _ArrayLikeWrapper(np.array([1, 2, 3]))

        Pass the object to functions with `()` instead of passing it
        directly as a variable, e.g.:

        >>> _validation.check_real(wrapped())

        """


class _NumpyArrayWrapper(_ArrayLikeWrapper[NumberType]):
    _array: NumpyArray[NumberType]
    dtype: np.dtype[NumberType]

    def all_func(self, func: Callable[..., Any], *args):
        return np.all(func(self._array, *args))

    def __call__(self) -> NumpyArray[NumberType]:
        return self  # type: ignore[return-value]

    def as_iterable(self) -> Iterable[NumberType]:
        return self._array.ravel(order='A')

    def to_list(
        self,
        input_array: _ArrayLikeOrScalar[NumberType],
        copy: bool,
    ) -> Union[NumberType, _FiniteNestedList[NumberType]]:
        return self._array.tolist()

    def to_tuple(
        self,
        input_array: _ArrayLikeOrScalar[NumberType],
        copy: bool,
    ) -> Union[NumberType, _FiniteNestedTuple[NumberType]]:
        return _cast_to_tuple(self._array)

    def to_numpy(
        self,
        input_array: _ArrayLikeOrScalar[NumberType],
        copy: bool,
    ) -> NumpyArray[NumberType]:
        out = self._array
        return np.ndarray.copy(out) if copy and out is input_array else out

    def change_dtype(self, dtype):
        self._array = self._array.astype(dtype, copy=False)


class _BuiltinWrapper(_ArrayLikeWrapper[NumberType]):
    def all_func(self, func: Callable[..., Any], *args):
        return all(func(x, *args) for x in self.as_iterable())

    def to_numpy(
        self,
        input_array: _ArrayLikeOrScalar[NumberType],
        copy: bool,
    ) -> NumpyArray[NumberType]:
        return np.array(self._array)


class _NumberWrapper(_BuiltinWrapper[NumberType]):
    _array: NumberType

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
    def dtype(self) -> Type[NumberType]:
        return type(self._array)

    def as_iterable(self) -> Iterable[NumberType]:
        return (self._array,)

    def __call__(self) -> NumberType:
        return self  # type: ignore[return-value]

    def to_list(self, input_array: _ArrayLikeOrScalar[NumberType], copy: bool) -> NumberType:
        return self._array

    def to_tuple(self, input_array: _ArrayLikeOrScalar[NumberType], copy: bool) -> NumberType:
        return self._array

    def change_dtype(self, dtype):
        if self.dtype is dtype:
            return
        else:
            self._array = dtype(self._array)


class _SequenceWrapper(_BuiltinWrapper[NumberType]):
    _array: Sequence[NumberType]

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
    def dtype(self) -> Type[NumberType]:
        self._dtype: Type[NumberType]
        if self._dtype is None:
            self._dtype = _get_dtype_from_iterable(self.as_iterable())
        return self._dtype

    def as_iterable(self) -> Iterable[NumberType]:
        return self._array

    def __call__(self) -> Sequence[NumberType]:
        return self  # type: ignore[return-value]

    def to_list(self, input_array: _ArrayLikeOrScalar[NumberType], copy: bool) -> List[NumberType]:
        out = self._array if isinstance(self._array, list) else list(self._array)
        return deepcopy(out) if copy and out is input_array else out

    def to_tuple(
        self,
        input_array: _ArrayLikeOrScalar[NumberType],
        copy: bool,
    ) -> Tuple[NumberType, ...]:
        out = self._array if isinstance(self._array, tuple) else tuple(self._array)
        return deepcopy(out) if copy and out is input_array else out

    def change_dtype(self, dtype):
        self._array = self._array.__class__(dtype(x) for x in self._array)
        self._dtype = dtype


class _NestedSequenceWrapper(_BuiltinWrapper[NumberType]):
    _array: Sequence[Sequence[NumberType]]

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
    def dtype(self) -> Type[NumberType]:
        self._dtype: Type[NumberType]
        if self._dtype is None:
            self._dtype = _get_dtype_from_iterable(self.as_iterable())
        return self._dtype

    def as_iterable(self) -> Iterable[NumberType]:
        return itertools.chain.from_iterable(self._array)

    def __call__(self) -> Sequence[Sequence[NumberType]]:
        return self  # type: ignore[return-value]

    def to_list(
        self,
        input_array: _ArrayLikeOrScalar[NumberType],
        copy: bool,
    ) -> List[List[NumberType]]:
        return (
            self._array
            if isinstance(self._array, list) and not copy
            else [list(sub_array) for sub_array in self._array]
        )

    def to_tuple(
        self,
        input_array: _ArrayLikeOrScalar[NumberType],
        copy: bool,
    ) -> Tuple[Tuple[NumberType, ...]]:
        return (
            self._array
            if isinstance(self._array, tuple)
            else tuple(tuple(sub_array) for sub_array in self._array)
        )

    def change_dtype(self, dtype):
        out = self._array.__class__(
            sub_array.__class__(dtype(item) for item in sub_array) for sub_array in self._array
        )
        self._array = out
        self._dtype = dtype


def _get_dtype_from_iterable(iterable: Iterable[NumberType]):
    # Note: This function assumes all elements are numeric.

    # create a set with all dtypes
    # exit early if float
    dtypes = set()
    for element in iterable:
        dtype = type(element)
        if dtype is float:
            return cast(Type[NumberType], float)
        else:
            dtypes.add(dtype)
    if len(dtypes) == 0:
        return cast(Type[NumberType], float)
    elif dtypes in [{int}, {bool, int}]:
        return cast(Type[NumberType], int)
    elif dtypes == {bool}:
        return cast(Type[NumberType], bool)
    else:
        raise TypeError(  # pragma: no cover
            f"Unexpected error: dtype should be numeric, got {dtypes} instead.",
        )
