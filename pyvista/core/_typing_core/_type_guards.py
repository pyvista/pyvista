"""Type guards for checking array-like type definitions."""

from typing import Any, Iterable, Sequence, Tuple, Type, TypeVar, cast

import numpy as np
from typing_extensions import TypeGuard

from ._array_like import _ArrayLikeOrScalar, _NumberType

_BuiltinNumberType = TypeVar('_BuiltinNumberType', float, int, bool)


def _is_Number(array: _ArrayLikeOrScalar[_NumberType]) -> TypeGuard[_NumberType]:
    return isinstance(array, (float, int, bool, np.floating, np.integer, np.bool_))


def _is_NumberSequence(
    array: _ArrayLikeOrScalar[_NumberType],
) -> TypeGuard[Sequence[_BuiltinNumberType]]:
    return isinstance(array, (tuple, list)) and _has_element_types(array, (float, int))


def _is_NestedNumberSequence(array: _ArrayLikeOrScalar[_NumberType]) -> bool:
    if (
        isinstance(array, (tuple, list))
        and len(array) > 0
        and all(_is_NumberSequence(subarray) for subarray in array)
    ):
        # We have the correct type, now check all subarray shapes are equal
        array = cast(Sequence[Sequence[_NumberType]], array)
        sub_shape = len(array[0])
        return all(len(sub_array) == sub_shape for sub_array in array)
    return False


def _has_element_types(array: Iterable[Any], types: Tuple[Type[Any], ...]) -> bool:
    """Check that iterable elements have the specified type."""
    return all(isinstance(item, types) for item in array)
