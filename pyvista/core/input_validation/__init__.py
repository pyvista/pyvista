"""Input validation functions."""

import sys

if sys.version_info >= (3, 9):
    from .check import check_is_subdtype  # noqa: 401
else:
    # handle TypeError: 'ABCMeta' object is not subscriptable
    # for type Sequence[DTypeLike]
    from .check import _check_is_subdtype_legacy as check_is_subdtype  # noqa: 401

from .check import (  # noqa: 401
    check_has_length,
    check_has_shape,
    check_is_finite,
    check_is_greater_than,
    check_is_in_range,
    check_is_instance,
    check_is_integerlike,
    check_is_iterable,
    check_is_iterable_of_some_type,
    check_is_iterable_of_strings,
    check_is_less_than,
    check_is_nonnegative,
    check_is_number,
    check_is_real,
    check_is_scalar,
    check_is_sequence,
    check_is_sorted,
    check_is_string,
    check_is_string_in_iterable,
    check_is_type,
)
from .validate import (  # noqa: 401
    validate_array,
    validate_array3,
    validate_arrayN,
    validate_arrayN_uintlike,
    validate_arrayNx3,
    validate_axes,
    validate_data_range,
    validate_number,
    validate_transform3x3,
    validate_transform4x4,
)
