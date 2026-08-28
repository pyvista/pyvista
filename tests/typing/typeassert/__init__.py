"""Assert that a value's static type and its runtime type agree.

A case file is a stack of one-line `assert_types(expression, ExpectedType)`
calls. Mypy checks the left of each pair against the right exactly; running the
line checks the value the expression actually produces. `collect_cases` turns
each line into its own test, and `run_mypy` supplies the static half.

Nothing here is specific to the project under test.
"""

from __future__ import annotations

from ._assertions import assert_types
from ._cases import Case
from ._cases import CaseFile
from ._cases import collect_cases
from ._mypy import MypyError
from ._mypy import run_mypy

__all__ = ['Case', 'CaseFile', 'MypyError', 'assert_types', 'collect_cases', 'run_mypy']
