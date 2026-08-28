"""The one assertion a typing case makes."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    # Mypy resolves an aliased import back to its original definition, so it applies
    # its `assert_type` special case here: the expression's inferred type must match
    # `expected` *exactly*, not merely be assignable to it. At runtime the definition
    # below runs instead and checks the value. One call, both halves.
    from typing_extensions import assert_type as assert_types
else:
    from pycroscope.checker import Checker
    from pycroscope.runtime import CanAssignError
    from pycroscope.runtime import KnownValue
    from pycroscope.runtime import Relation
    from pycroscope.runtime import has_relation
    from pycroscope.runtime import type_from_runtime

    @functools.cache
    def _checker() -> Checker:
        """Return the shared checker, built on first use."""
        return Checker()

    def assert_types(value: object, expected: Any) -> object:
        """Assert `value` is assignable to `expected` at runtime, and return it."""
        # pycroscope's own `get_assignability_error` memoises against a module-global
        # checker, which keeps every checked value alive. Use our own so the memo can
        # be dropped -- PyVista's suite fails a test that leaks a VTK object.
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
            # An assertion that failed, not a caller passing the wrong kind of argument.
            raise AssertionError(msg)  # noqa: TRY004
        return value
