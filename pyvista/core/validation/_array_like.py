"""Functions for processing array-like inputs."""
from __future__ import annotations

from typing import Generic, List, Sequence, Tuple, Type, TypeVar, Union, cast, overload

import numpy as np

from pyvista.core._typing_core import NumpyArray
from pyvista.core._typing_core._array_like import (
    _ArrayLikeOrScalar,
    _NumberSequence,
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
_NumberType_co = TypeVar('_NumberType_co', bound=Union[float, int, np.number], covariant=True)
DTypeLike = Union[np.dtype[_NumberType_co], Type[_NumberType_co], str]


class _ArrayLikeWrapper(Generic[_NumberType]):
    __slots__ = ('array', 'dtype', 'shape', 'ndim')
    array: _ArrayLikeOrScalar[_NumberType]
    dtype: Type[_NumberType]
    shape: Shape
    ndim: int

    @overload
    def __new__(  # type: ignore[overload-overlap]
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
    def __new__(  # type: ignore[overload-overlap]
        cls,
        array: _NumpyArraySequence[_NumberType],
    ) -> _NumpyArraySequenceWrapper[_NumberType]:
        ...

    @overload
    def __new__(
        cls,
        array: _NumberSequence[_NumberType],
    ) -> _NumberSequenceWrapper[_NumberType]:
        ...

    def __new__(
        cls,
        array: _ArrayLikeOrScalar[_NumberType],
    ):
        """Return ArrayLike shape, dtype, and ndim."""
        # Note: This implementation is purposefully verbose and explicit
        # so that mypy can correctly infer the return types.

        shape: List[int] = []
        depth: int = 0

        def _not_empty():
            return shape[-1] > 0

        if isinstance(array, Sequence):
            depth = 1
            shape.append(len(array))
            if _not_empty() and isinstance(array[0], Sequence):
                sub_shape: Union[Tuple[()], Tuple[int, ...], int]
                depth = 2
                shape.append(len(array[0]))
                if _not_empty() and isinstance(array[0][0], Sequence):
                    depth = 3
                    shape.append(len(array[0][0]))
                    if _not_empty() and isinstance(array[0][0][0], Sequence):
                        depth = 4
                        shape.append(len(array[0][0][0]))
                        if _not_empty() and isinstance(array[0][0][0][0], Sequence):
                            raise TypeError(
                                "Nested sequences with more than 4 levels are not supported."
                            )

                        elif _not_empty() and isinstance(array[0][0][0][0], np.ndarray):
                            # 4D sequence of numpy arrays
                            sub_shape = array[0][0][0][0].shape
                            _check_all_subarray_shapes_are_equal(
                                cast(Sequence[NumpyArray[_NumberType]], array[0][0][0]),
                                sub_shape,
                            )
                            shape.extend(sub_shape)

                            numpy_sequence_wrapper4 = object.__new__(_NumpyArraySequenceWrapper)
                            numpy_sequence_wrapper4.__setattr__('array', array)
                            numpy_sequence_wrapper4.__setattr__('shape', tuple(shape))
                            numpy_sequence_wrapper4.__setattr__(
                                'dtype', array[0][0][0][0].dtype.type
                            )
                            numpy_sequence_wrapper4.__setattr__('ndim', len(shape))
                            return numpy_sequence_wrapper4

                        else:
                            # 4D sequence of numbers
                            sub_shape = shape[3]
                            _check_all_subarray_shapes_are_equal(
                                cast(Sequence[Sequence[_NumberType]], array[0][0]), sub_shape
                            )
                            try:
                                dtype = cast(Type[_NumberType], type(array[0][0][0][0]))
                            except IndexError:
                                dtype = cast(Type[_NumberType], float)
                            num_sequence_wrapper4 = object.__new__(_NumberSequenceWrapper)
                            num_sequence_wrapper4.__setattr__('array', array)
                            num_sequence_wrapper4.__setattr__('shape', tuple(shape))
                            num_sequence_wrapper4.__setattr__('dtype', dtype)
                            num_sequence_wrapper4.__setattr__('ndim', depth)
                            return num_sequence_wrapper4

                    elif _not_empty() and isinstance(array[0][0][0], np.ndarray):
                        # 3D sequence of numpy arrays
                        sub_shape = array[0][0][0].shape
                        _check_all_subarray_shapes_are_equal(
                            cast(Sequence[NumpyArray[_NumberType]], array[0][0]), sub_shape
                        )
                        shape.extend(sub_shape)

                        numpy_sequence_wrapper3 = object.__new__(_NumpyArraySequenceWrapper)
                        numpy_sequence_wrapper3.__setattr__('array', array)
                        numpy_sequence_wrapper3.__setattr__('shape', tuple(shape))
                        numpy_sequence_wrapper3.__setattr__('dtype', array[0][0][0].dtype.type)
                        numpy_sequence_wrapper3.__setattr__('ndim', len(shape))
                        return numpy_sequence_wrapper3

                    else:
                        # 3D sequence of numbers
                        sub_shape = shape[2]
                        _check_all_subarray_shapes_are_equal(
                            cast(Sequence[Sequence[_NumberType]], array[0]), sub_shape
                        )
                        try:
                            dtype = cast(Type[_NumberType], type(array[0][0][0]))
                        except IndexError:
                            dtype = cast(Type[_NumberType], float)
                        num_sequence_wrapper3 = object.__new__(_NumberSequenceWrapper)
                        num_sequence_wrapper3.__setattr__('array', array)
                        num_sequence_wrapper3.__setattr__('shape', tuple(shape))
                        num_sequence_wrapper3.__setattr__('dtype', dtype)
                        num_sequence_wrapper3.__setattr__('ndim', depth)
                        return num_sequence_wrapper3

                elif _not_empty() and isinstance(array[0][0], np.ndarray):
                    # 2D sequence of numpy arrays
                    sub_shape = array[0][0].shape
                    _check_all_subarray_shapes_are_equal(
                        cast(Sequence[NumpyArray[_NumberType]], array[0]), sub_shape
                    )
                    shape.extend(sub_shape)

                    numpy_sequence_wrapper2 = object.__new__(_NumpyArraySequenceWrapper)
                    numpy_sequence_wrapper2.__setattr__('array', array)
                    numpy_sequence_wrapper2.__setattr__('shape', tuple(shape))
                    numpy_sequence_wrapper2.__setattr__('dtype', array[0][0].dtype.type)
                    numpy_sequence_wrapper2.__setattr__('ndim', len(shape))
                    return numpy_sequence_wrapper2
                else:
                    # 2D sequence of numbers
                    sub_shape = shape[1]
                    _check_all_subarray_shapes_are_equal(
                        cast(Sequence[Sequence[_NumberType]], array), sub_shape
                    )
                    try:
                        dtype = cast(Type[_NumberType], type(array[0][0]))
                    except IndexError:
                        dtype = cast(Type[_NumberType], float)

                    num_sequence_wrapper2 = object.__new__(_NumberSequenceWrapper)
                    num_sequence_wrapper2.__setattr__('array', array)
                    num_sequence_wrapper2.__setattr__('shape', tuple(shape))
                    num_sequence_wrapper2.__setattr__('dtype', dtype)
                    num_sequence_wrapper2.__setattr__('ndim', depth)
                    return num_sequence_wrapper2

            elif _not_empty() and isinstance(array[0], np.ndarray):
                # 1D sequence of numpy arrays
                sub_shape = array[0].shape
                _check_all_subarray_shapes_are_equal(
                    cast(Sequence[NumpyArray[_NumberType]], array), sub_shape
                )
                shape.extend(array[0].shape)

                numpy_sequence_wrapper1 = object.__new__(_NumpyArraySequenceWrapper)
                numpy_sequence_wrapper1.__setattr__('array', array)
                numpy_sequence_wrapper1.__setattr__('shape', tuple(shape))
                numpy_sequence_wrapper1.__setattr__('dtype', array[0].dtype.type)
                numpy_sequence_wrapper1.__setattr__('ndim', len(shape))
                return numpy_sequence_wrapper1
            else:
                # 1D sequence of numbers
                try:
                    dtype = cast(Type[_NumberType], type(array[0]))
                except IndexError:
                    dtype = cast(Type[_NumberType], float)
                num_sequence_wrapper1 = object.__new__(_NumberSequenceWrapper)
                num_sequence_wrapper1.__setattr__('array', array)
                num_sequence_wrapper1.__setattr__('shape', tuple(shape))
                num_sequence_wrapper1.__setattr__('dtype', dtype)
                num_sequence_wrapper1.__setattr__('ndim', depth)
                return num_sequence_wrapper1

        elif isinstance(array, np.ndarray):
            # non-nested numpy array
            numpy_wrapper = object.__new__(_NumpyArrayWrapper)
            numpy_wrapper.__setattr__('array', array)
            numpy_wrapper.__setattr__('shape', array.shape)
            numpy_wrapper.__setattr__('dtype', array.dtype.type)
            numpy_wrapper.__setattr__('ndim', array.ndim)
            return numpy_wrapper

        else:
            # just a number/scalar type
            scalar_wrapper = object.__new__(_ScalarWrapper)
            scalar_wrapper.__setattr__('array', array)
            scalar_wrapper.__setattr__('shape', tuple(shape))
            scalar_wrapper.__setattr__('dtype', type(array))
            scalar_wrapper.__setattr__('ndim', depth)
            return scalar_wrapper


from typing import Any, Protocol, runtime_checkable

_DType_co = TypeVar('_DType_co', bound=np.generic, covariant=True)


@runtime_checkable
class _SupportsArray(Protocol[_DType_co]):
    def __array__(self) -> np.ndarray[Any, np.dtype[_DType_co]]:
        ...


class _NumpyArrayWrapper(_ArrayLikeWrapper[_NumberType]):
    array: NumpyArray[_NumberType]


class _ScalarWrapper(_ArrayLikeWrapper[_NumberType]):
    array: _NumberType


class _NumberSequenceWrapper(_ArrayLikeWrapper[_NumberType]):
    array: _NumberSequence[_NumberType]


class _NumpyArraySequenceWrapper(_ArrayLikeWrapper[_NumberType]):
    array: _NumpyArraySequence[_NumberType]


# reveal_type(ArrayLikeWrapper(1))
# reveal_type(ArrayLikeWrapper(np.array([1], dtype=int)))
# reveal_type(ArrayLikeWrapper(np.array([1], dtype=int)).array)
# reveal_type(ArrayLikeWrapper(1).array)
# reveal_type(ArrayLikeWrapper(1).dtype)
# reveal_type(ArrayLikeWrapper([1]).array)
# reveal_type(ArrayLikeWrapper([[1]]))

# props0 = _array_like_props(0)
# reveal_type(props0[0])
# reveal_type(props0[1])
# reveal_type(props0[2])
# reveal_type(props0[3])
#
# props0n = _array_like_props(np.array([1], dtype=int))
# reveal_type(props0n[0])
# reveal_type(props0n[1])
# reveal_type(props0n[2])
# reveal_type(props0n[3])
#
# props1 = _array_like_props([0])
# reveal_type(props1[0])
# reveal_type(props1[1])
# reveal_type(props1[2])
# reveal_type(props1[3])
#
# reveal_type([np.array(1, dtype=int)])
# props1n = _array_like_props([np.array(1, dtype=int)])
# reveal_type(props1n[0])
# reveal_type(props1n[1])
# reveal_type(props1n[2])
# reveal_type(props1n[3])

_SequenceArgType = TypeVar('_SequenceArgType', Sequence[Sequence], Sequence[np.ndarray])


def _check_all_subarray_shapes_are_equal(array: _SequenceArgType, sub_shape) -> None:
    # Type annotations cannot infer the shape or length of sub-arrays,
    # so we check this at runtime
    if isinstance(array[0], Sequence):
        all_same = all(len(x) == sub_shape for x in array)
    else:
        all_same = all(x.shape == sub_shape for x in array)

    if not all_same:
        raise ValueError(
            "The nested sequence array has an inhomogeneous shape. "
            "All sub-arrays must have the same shape."
        )


# def wrap_arraylike(array: _ArrayLikeOrScalar):
#     return ArrayLikeWrapper(array)

#
# T = TypeVar('T', bound=int)
# class Factory(Generic[T]):
#
#     @overload
#     def __new__(cls,
#         value: List[T]
#     ) -> GenericList[T]:
#         ...
#
#     @overload
#     def __new__(cls,
#         value: T
#     ) -> GenericType[T]:
#         ...
#     def __new__(cls, value):
#         if isinstance(value, List):
#             return object.__new__(GenericList[T])
#         else:
#             return object.__new__(GenericType[T])
# class GenericList(Factory[T]):
#     ...
#
# class GenericType(Factory[T]):
#     ...
#
# reveal_type(Factory([1]))
