"""Type guards for checking array-like type definitions."""

from __future__ import annotations

from itertools import islice
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import cast

try:
    from typing import TypeGuard
except ImportError:
    from typing_extensions import TypeGuard

import numpy as np

from ._array_like import NumberType
from ._array_like import _PyNumberType

if TYPE_CHECKING:
    from ._aliases import _ArrayLikeOrScalar


def _is_Number(array: _ArrayLikeOrScalar[NumberType]) -> TypeGuard[NumberType]:
    return isinstance(array, (float, int, bool, np.floating, np.integer, np.bool_))


def _is_NumberSequence(
    array: _ArrayLikeOrScalar[NumberType],
) -> TypeGuard[Sequence[_PyNumberType]]:
    return isinstance(array, (tuple, list)) and _has_element_types(array, (float, int))


def _is_NestedNumberSequence(array: _ArrayLikeOrScalar[NumberType]) -> bool:
    if (
        isinstance(array, (tuple, list))
        and len(array) > 0
        and all(_is_NumberSequence(subarray) for subarray in array)
    ):
        # We have the correct type, now check all subarray shapes are equal
        array = cast(Sequence[Sequence[NumberType]], array)
        sub_shape = len(array[0])
        return all(len(sub_array) == sub_shape for sub_array in array)
    return False


def _has_element_types(array: Iterable[Any], types: Tuple[Type[Any], ...], N=100) -> bool:
    """Check that iterable elements have the specified type.

    Parameters
    ----------
    N : int, default: 100
        Only check the first `N` elements. Can be used to reduce the
        performance cost of this check. Set to `None` to check all elements.
    """
    iterator = islice(array, N)
    return all(isinstance(item, types) for item in iterator)
