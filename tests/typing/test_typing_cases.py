"""Run PyVista's typing cases, statically and at runtime.

Each `assert_types(expression, ExpectedType)` line in ``tests/typing/cases``
becomes two tests: one that runs the line and checks the value it produces, and
one that checks what Mypy inferred for the same line. See ``typeassert`` for
the machinery, and ``CONTRIBUTING.rst`` for how to write a case.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import PYVISTA_ROOT_DIR
from tests.typing.typeassert import Case
from tests.typing.typeassert import CaseFile
from tests.typing.typeassert import collect_cases
from tests.typing.typeassert import run_mypy

CASES_DIR = Path(__file__).parent / 'cases'
CASES_PACKAGE = '.'.join(CASES_DIR.relative_to(PYVISTA_ROOT_DIR).parts)


@pytest.fixture(scope='session')
def mypy_diagnostics(worker_id):
    """Type-check the cases once per session and return the errors keyed by file."""
    # A cache dir per xdist worker: the run is cheap once warm, and sharing one
    # cache between concurrent workers is what makes it stale.
    cache_dir = PYVISTA_ROOT_DIR / '.mypy_cache' / f'typing-cases-{worker_id}'
    return run_mypy(CASES_PACKAGE, root=PYVISTA_ROOT_DIR, cache_dir=cache_dir)


def pytest_generate_tests(metafunc):
    """Generate one test per case, and one per case file for its setup lines."""
    files = collect_cases(CASES_DIR)
    if 'case' in metafunc.fixturenames:
        cases = [(file, case) for file in files for case in file.cases]
        ids = [f'{file.name}: {case.id}' for file, case in cases]
        metafunc.parametrize(('case_file', 'case'), cases, ids=ids)
    if 'case_file_setup' in metafunc.fixturenames:
        metafunc.parametrize('case_file_setup', files, ids=[file.name for file in files])


def test_runtime_type(case_file: CaseFile, case: Case) -> None:
    """Assert the value this case builds has the type the case expects."""
    del case_file
    case.run()


def test_static_type(case_file: CaseFile, case: Case, mypy_diagnostics) -> None:
    """Assert Mypy infers the type this case expects."""
    errors = [
        diagnostic
        for diagnostic in mypy_diagnostics.get(case_file.path.resolve(), [])
        if diagnostic.line in case.lines
    ]
    if errors:
        _fail(case_file, errors)


def test_case_file_setup(case_file_setup: CaseFile, request) -> None:
    """Assert the file is well formed and Mypy reports nothing outside its cases."""
    # Report the file's own problem before asking for a type-check of it: a file Mypy
    # cannot parse fails the whole run, which would otherwise mask the reason.
    if case_file_setup.error is not None:
        pytest.fail(case_file_setup.error)
    mypy_diagnostics = request.getfixturevalue('mypy_diagnostics')
    errors = [
        diagnostic
        for diagnostic in mypy_diagnostics.get(case_file_setup.path.resolve(), [])
        if diagnostic.line in case_file_setup.setup_lines
    ]
    if errors:
        _fail(case_file_setup, errors)


def _fail(case_file: CaseFile, errors) -> None:
    """Fail with Mypy's messages against the source lines they came from."""
    source = case_file.path.read_text(encoding='utf-8').splitlines()
    report = '\n'.join(
        f'{case_file.name}:{error.line}: {error.message}\n\t{source[error.line - 1].strip()}'
        for error in errors
    )
    pytest.fail(f'Mypy reported {len(errors)} error(s):\n{report}')
