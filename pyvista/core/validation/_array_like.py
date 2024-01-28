"""Functions for processing array-like inputs."""
from collections import namedtuple
from typing import List, Literal, Sequence, Tuple, Type, TypeVar, Union, cast, overload

import numpy as np

from pyvista.core._typing_core import NumpyArray
from pyvista.core._typing_core._array_like import (
    _ArrayLikeOrScalar,
    _NumberSequence1D,
    _NumberSequence2D,
    _NumberSequence3D,
    _NumberSequence4D,
    _NumberType,
    _NumpyArraySequence1D,
    _NumpyArraySequence2D,
    _NumpyArraySequence3D,
    _NumpyArraySequence4D,
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
DTypeLike = Union[np.dtype, Type]


# Store array-like props using named tuples
# Use different names for different array types
_ScalarDTypeTuple = namedtuple('_ScalarDTypeTuple', ['array', 'shape', 'dtype', 'ndim'])
_NumpyArrayTuple = namedtuple('_NumpyArrayTuple', ['array', 'shape', 'dtype', 'ndim'])
_NumberSequenceTuple = namedtuple('_NumberSequenceTuple', ['array', 'shape', 'dtype', 'ndim'])
_NumpyArraySequenceTuple = namedtuple(
    '_NumpyArraySequenceTuple', ['array', 'shape', 'dtype', 'ndim']
)


@overload
def _array_like_props(
    array: _NumberType,
) -> Tuple[_NumberType, Tuple[()], Type[_NumberType], Literal[0],]:
    ...


@overload
def _array_like_props(
    array: NumpyArray[_NumberType],
) -> Tuple[NumpyArray[_NumberType], Union[Tuple[()], Tuple[int, ...]], Type[_NumberType], int,]:
    ...


@overload
def _array_like_props(
    array: _NumpyArraySequence1D[_NumberType],
) -> Tuple[_NumpyArraySequence1D[_NumberType], Tuple[int, ...], Type[_NumberType], int,]:
    ...


@overload
def _array_like_props(
    array: _NumpyArraySequence2D[_NumberType],
) -> Tuple[_NumpyArraySequence2D[_NumberType], Tuple[int, ...], Type[_NumberType], int,]:
    ...


@overload
def _array_like_props(
    array: _NumpyArraySequence3D[_NumberType],
) -> Tuple[_NumpyArraySequence3D[_NumberType], Tuple[int, ...], Type[_NumberType], int,]:
    ...


@overload
def _array_like_props(
    array: _NumpyArraySequence4D[_NumberType],
) -> Tuple[_NumpyArraySequence4D[_NumberType], Tuple[int, ...], Type[_NumberType], int,]:
    ...


@overload
def _array_like_props(
    array: _NumberSequence4D[_NumberType],
) -> Tuple[
    _NumberSequence4D[_NumberType],
    Tuple[int, int, int, int],
    Type[_NumberType],
    Literal[4],
]:
    ...


@overload
def _array_like_props(
    array: _NumberSequence3D[_NumberType],
) -> Tuple[_NumberSequence3D[_NumberType], Tuple[int, int, int], Type[_NumberType], Literal[3],]:
    ...


@overload
def _array_like_props(
    array: _NumberSequence2D[_NumberType],
) -> Tuple[_NumberSequence2D[_NumberType], Tuple[int, int], Type[_NumberType], Literal[2],]:
    ...


@overload
def _array_like_props(
    array: _NumberSequence1D[_NumberType],
) -> Tuple[_NumberSequence1D[_NumberType], Tuple[int], Type[_NumberType], Literal[1],]:
    ...


def _array_like_props(
    array: _ArrayLikeOrScalar[_NumberType],
) -> Tuple[_ArrayLikeOrScalar[_NumberType], Tuple[int, ...], Type[_NumberType], int]:
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
                        _check_all_subarray_shapes(
                            cast(Sequence[NumpyArray[_NumberType]], array[0][0][0]),
                            sub_shape,
                        )
                        shape.extend(sub_shape)
                        return _NumpyArraySequenceTuple(
                            array=array,
                            shape=tuple(shape),
                            dtype=array[0][0][0][0].dtype.type,
                            ndim=len(shape),
                        )
                    else:
                        # 4D sequence of numbers
                        sub_shape = shape[3]
                        _check_all_subarray_shapes(
                            cast(Sequence[Sequence[_NumberType]], array[0][0]), sub_shape
                        )
                        try:
                            dtype = cast(Type[_NumberType], type(array[0][0][0][0]))
                        except IndexError:
                            dtype = cast(Type[_NumberType], float)
                        return _NumberSequenceTuple(
                            array=array, shape=tuple(shape), dtype=dtype, ndim=depth
                        )

                elif _not_empty() and isinstance(array[0][0][0], np.ndarray):
                    # 3D sequence of numpy arrays
                    sub_shape = array[0][0][0].shape
                    _check_all_subarray_shapes(
                        cast(Sequence[NumpyArray[_NumberType]], array[0][0]), sub_shape
                    )
                    shape.extend(sub_shape)
                    return _NumpyArraySequenceTuple(
                        array=array,
                        shape=tuple(shape),
                        dtype=array[0][0][0].dtype.type,
                        ndim=len(shape),
                    )
                else:
                    # 3D sequence of numbers
                    sub_shape = shape[2]
                    _check_all_subarray_shapes(
                        cast(Sequence[Sequence[_NumberType]], array[0]), sub_shape
                    )
                    try:
                        dtype = cast(Type[_NumberType], type(array[0][0][0]))
                    except IndexError:
                        dtype = cast(Type[_NumberType], float)
                    return _NumberSequenceTuple(
                        array=array, shape=tuple(shape), dtype=dtype, ndim=depth
                    )

            elif _not_empty() and isinstance(array[0][0], np.ndarray):
                # 2D sequence of numpy arrays
                sub_shape = array[0][0].shape
                _check_all_subarray_shapes(
                    cast(Sequence[NumpyArray[_NumberType]], array[0]), sub_shape
                )
                shape.extend(sub_shape)
                return _NumpyArraySequenceTuple(
                    array=array, shape=tuple(shape), dtype=array[0][0].dtype.type, ndim=len(shape)
                )
            else:
                # 2D sequence of numbers
                sub_shape = shape[1]
                _check_all_subarray_shapes(cast(Sequence[Sequence[_NumberType]], array), sub_shape)
                try:
                    dtype = cast(Type[_NumberType], type(array[0][0]))
                except IndexError:
                    dtype = cast(Type[_NumberType], float)
                return _NumberSequenceTuple(
                    array=array, shape=tuple(shape), dtype=dtype, ndim=depth
                )

        elif _not_empty() and isinstance(array[0], np.ndarray):
            # 1D sequence of numpy arrays
            sub_shape = array[0].shape
            _check_all_subarray_shapes(cast(Sequence[NumpyArray[_NumberType]], array), sub_shape)
            shape.extend(array[0].shape)
            return _NumpyArraySequenceTuple(
                array=array, shape=tuple(shape), dtype=array[0].dtype.type, ndim=len(shape)
            )
        else:
            # 1D sequence of numbers
            try:
                dtype = cast(Type[_NumberType], type(array[0]))
            except IndexError:
                dtype = cast(Type[_NumberType], float)
            return _NumberSequenceTuple(array=array, shape=tuple(shape), dtype=dtype, ndim=depth)

    elif isinstance(array, np.ndarray):
        # non-nested numpy array
        return _NumpyArrayTuple(
            array=array,
            shape=array.shape,
            dtype=array.dtype.type,
            ndim=array.ndim,
        )
    else:
        # just a number/scalar type
        return _ScalarDTypeTuple(array=array, shape=tuple(shape), dtype=type(array), ndim=depth)


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


def _check_all_subarray_shapes(array: _SequenceArgType, sub_shape) -> None:
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
