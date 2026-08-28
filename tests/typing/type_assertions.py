"""Runtime counterpart to :func:`typing_extensions.assert_type`.

Test cases under ``tests/typing/cases`` pair the two: ``assert_type`` pins the
type Mypy infers, ``assert_runtime_type`` pins the type the value actually has.
See ``tests/typing/test_static_types.py`` for how the pairing is enforced.
"""

from __future__ import annotations

import functools
from typing import Any

from pycroscope.checker import Checker
from pycroscope.runtime import CanAssignError
from pycroscope.runtime import KnownValue
from pycroscope.runtime import Relation
from pycroscope.runtime import has_relation
from pycroscope.runtime import type_from_runtime

__all__ = ['assert_runtime_type']


@functools.cache
def _checker() -> Checker:
    """Return the shared checker, built on first use."""
    return Checker()


def assert_runtime_type(value: object, expected: Any) -> None:
    """Assert `value` is assignable to the `expected` type at runtime."""
    # pycroscope's own `get_assignability_error` memoizes against a module-global
    # checker, which keeps every checked value alive and trips the leak check in
    # `tests/conftest.py`. Use our own checker so the memo can be dropped.
    checker = _checker()
    try:
        relation = has_relation(
            type_from_runtime(expected), KnownValue(value), Relation.ASSIGNABLE, checker
        )
    finally:
        cache = checker.get_relation_cache()
        if cache is not None:
            cache.clear()

    if isinstance(relation, CanAssignError):
        msg = (
            f'Runtime value of type {type(value).__name__!r} is not assignable '
            f'to the expected type:\n\t{expected}\n\n{relation.display(depth=0)}'
        )
        raise AssertionError(msg)  # noqa: TRY004  # a failed assertion, not a bad argument
