"""Type guards for checking array-like type definitions."""

from itertools import islice
from typing import Any, Iterable, Sequence, Tuple, Type, TypeVar, cast

import numpy as np

from ._array_like import _ArrayLikeOrScalar, _NumberType

_BuiltinNumberType = TypeVar('_BuiltinNumberType', float, int, bool, covariant=True)

# TODO: import and return TypeGuard instead of bool for python >= 3.10
#  (requires typing_extensions for python < 3.10)


def _is_Number(array: _ArrayLikeOrScalar[_NumberType]) -> bool:
    return isinstance(array, (float, int, bool, np.floating, np.integer, np.bool_))


def _is_NumberSequence(array: _ArrayLikeOrScalar[_NumberType], N=None) -> bool:
    return isinstance(array, (tuple, list)) and _has_element_types(array, (float, int), N=N)


def _is_NestedNumberSequence(array: _ArrayLikeOrScalar[_NumberType], N=None) -> bool:
    if (
        isinstance(array, (tuple, list))
        and len(array) > 0
        and all(_is_NumberSequence(subarray, N=N) for subarray in array)
    ):
        # We have the correct type, now check all subarray shapes are equal
        array = cast(Sequence[Sequence[_NumberType]], array)
        sub_shape = len(array[0])
        return all(len(sub_array) == sub_shape for sub_array in array)
    return False


def _has_element_types(array: Iterable[Any], types: Tuple[Type[Any], ...], N=None) -> bool:
    """Check that iterable elements have the specified type.

    Parameters
    ----------
    N : int
        Only check the first `N` elements. Can be used to reduce the
        performance cost of this check. Set to `None` to check all elements.
    """
    iterator = islice(array, N)
    return all(isinstance(item, types) for item in iterator)
