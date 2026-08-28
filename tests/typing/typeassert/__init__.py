"""Assert that a value's static type and its runtime type agree.

A case file is a stack of one-line `assert_types(expression, ExpectedType)`
calls. A type checker checks the left of each pair against the right exactly;
running the line checks the value the expression actually produces. Wire the
pytest plugin up from a conftest next to the cases -- see `plugin`.

Nothing here is specific to the project under test.
"""

from __future__ import annotations

from ._assertions import assert_types
from ._cases import Case
from ._cases import CaseError
from ._cases import CaseFile
from ._cases import CaseSkipped
from ._cases import collect_case_file
from ._cases import collect_cases
from ._mypy import Diagnostic
from ._mypy import MypyError
from ._mypy import run_mypy
from .plugin import collect_cases_from

__all__ = [
    'Case',
    'CaseError',
    'CaseFile',
    'CaseSkipped',
    'Diagnostic',
    'MypyError',
    'assert_types',
    'collect_case_file',
    'collect_cases',
    'collect_cases_from',
    'run_mypy',
]
