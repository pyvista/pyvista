"""Check the static types asserted by the cases in ``tests/typing/cases``.

Each case file is an ordinary pytest module: its test functions build a value
and pin its type twice, once with :func:`typing_extensions.assert_type` for
Mypy and once with
:func:`~tests.typing.type_assertions.assert_runtime_type` for the interpreter.
Running the module under pytest checks the runtime half; this module runs Mypy
over the same files and checks the static half.

Mypy runs in a subprocess from a session-scoped fixture rather than at
collection time, so a Mypy failure fails these tests instead of aborting the
run. Diagnostics are attributed to the case function whose line span contains
them, which keeps the cases independent of each other and of their line
numbers.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
import re
import subprocess
import sys
from typing import NamedTuple

import pytest

from tests.conftest import PYVISTA_ROOT_DIR

CASES_DIR = Path(__file__).parent / 'cases'
CASES_PACKAGE = '.'.join(CASES_DIR.relative_to(PYVISTA_ROOT_DIR).parts)

# `path:line:col: severity: message`, with the column omitted for whole-file diagnostics.
_DIAGNOSTIC = re.compile(r'^(?P<path>.+?):(?P<line>\d+):(?:\d+:)? (?P<severity>\w+): (?P<msg>.*)$')


ASSERTIONS = ('assert_type', 'assert_runtime_type')


class _Assertion(NamedTuple):
    """One `assert_type` or `assert_runtime_type` call made by a case."""

    line: int
    kind: str
    target: str
    target_is_name: bool
    expected: str


class _Case(NamedTuple):
    """One case function, or the module scope of one case file."""

    path: Path
    name: str
    lines: frozenset[int]
    assertions: tuple[_Assertion, ...] = ()

    @property
    def id(self) -> str:
        """Return the test id for this case."""
        return f'{self.path.name}::{self.name}'


def _collect_assertions(node: ast.FunctionDef) -> tuple[_Assertion, ...]:
    """Return the type assertions made by one case function."""
    assertions = []
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id in ASSERTIONS
            and len(child.args) == 2
        ):
            target, expected = child.args
            assertions.append(
                _Assertion(
                    line=child.lineno,
                    kind=child.func.id,
                    target=ast.unparse(target),
                    target_is_name=isinstance(target, ast.Name),
                    # `unparse` normalises formatting, so two spellings of the same
                    # type expression compare equal and the text stays readable.
                    expected=ast.unparse(expected),
                )
            )
    return tuple(assertions)


def _collect_cases() -> list[_Case]:
    """Return every case function plus a module-scope case per file.

    Parsed with `ast` rather than imported, so collection stays independent of
    whether the cases themselves can be imported or type-checked.
    """
    cases = []
    for path in sorted(CASES_DIR.glob('test_*.py')):
        source = path.read_text(encoding='utf-8')
        tree = ast.parse(source, filename=str(path))
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_')
        ]
        for node in functions:
            lines = frozenset(range(node.lineno, (node.end_lineno or node.lineno) + 1))
            cases.append(_Case(path, node.name, lines, _collect_assertions(node)))
        # Whatever is left -- imports, fixtures, module constants -- is its own case,
        # so a broken import reads as such instead of failing every case in the file.
        covered = frozenset().union(*(case.lines for case in cases if case.path == path))
        module_lines = frozenset(range(1, len(source.splitlines()) + 1)) - covered
        if module_lines:
            cases.append(_Case(path, '<module>', module_lines))
    return cases


def _run_mypy(cache_dir: Path) -> str:
    """Type-check the case files and return Mypy's raw output."""
    # `--follow-imports=silent` types the PyVista symbols the cases use without
    # reporting PyVista's own diagnostics, which vary by platform and NumPy version.
    args = [
        sys.executable,
        '-m',
        'mypy',
        '--follow-imports=silent',
        '--no-color-output',
        '--no-error-summary',
        '--no-pretty',
        '--show-traceback',
        f'--cache-dir={cache_dir}',
        '--package',
        CASES_PACKAGE,
    ]
    process = subprocess.run(
        args, capture_output=True, cwd=PYVISTA_ROOT_DIR, text=True, check=False
    )
    if process.stderr:
        msg = f'Mypy failed to run:\n{" ".join(args)}\n\n{process.stderr}'
        raise RuntimeError(msg)
    return process.stdout


def _parse_diagnostics(output: str) -> dict[Path, list[tuple[int, str]]]:
    """Return Mypy's errors as `(line number, message)` keyed by resolved path."""
    diagnostics: dict[Path, list[tuple[int, str]]] = {}
    for line in output.splitlines():
        match = _DIAGNOSTIC.match(line)
        if match is None or match['severity'] != 'error':
            continue
        path = (PYVISTA_ROOT_DIR / match['path']).resolve()
        diagnostics.setdefault(path, []).append((int(match['line']), match['msg']))
    return diagnostics


@pytest.fixture(scope='session')
def mypy_diagnostics(worker_id) -> dict[Path, list[tuple[int, str]]]:
    """Run Mypy once per session and return its errors keyed by file."""
    # A cache dir per xdist worker: the run is cheap once warm, and sharing one
    # cache between concurrent workers is what makes it stale.
    cache_dir = PYVISTA_ROOT_DIR / '.mypy_cache' / f'typing-cases-{worker_id}'
    return _parse_diagnostics(_run_mypy(cache_dir))


def pytest_generate_tests(metafunc):
    """Generate one static test per case function."""
    if 'case' in metafunc.fixturenames:
        cases = _collect_cases()
        metafunc.parametrize('case', cases, ids=[case.id for case in cases])


def test_static_type(case: _Case, mypy_diagnostics) -> None:
    """Assert Mypy reports no error inside this case."""
    errors = [
        (line, msg)
        for line, msg in mypy_diagnostics.get(case.path.resolve(), [])
        if line in case.lines
    ]
    if errors:
        source = case.path.read_text(encoding='utf-8').splitlines()
        report = '\n'.join(
            f'{case.path.name}:{line}: {msg}\n\t{source[line - 1].strip()}' for line, msg in errors
        )
        pytest.fail(f'Mypy reported {len(errors)} error(s) in {case.name}:\n{report}')


def test_asserts_both_types(case: _Case) -> None:
    """Assert this case pins its type statically and at runtime, to the same type.

    Checked from the source rather than from Mypy, so a case that only pins one
    of the two -- and therefore proves nothing about the two agreeing -- is
    reported even when Mypy cannot run.
    """
    if case.name == '<module>':
        pytest.skip('Module scope holds no type assertions.')

    static = [assertion for assertion in case.assertions if assertion.kind == ASSERTIONS[0]]
    runtime = [assertion for assertion in case.assertions if assertion.kind == ASSERTIONS[1]]
    if not static:
        pytest.fail(f'{case.name} makes no `{ASSERTIONS[0]}` call.')

    positional = [assertion for assertion in case.assertions if not assertion.target_is_name]
    if positional:
        reported = '\n'.join(
            f'{case.path.name}:{assertion.line}: {assertion.kind}({assertion.target}, ...)'
            for assertion in positional
        )
        pytest.fail(
            f'Assert the type of a local name, so that both assertions describe the '
            f'same object rather than two separately evaluated expressions:\n{reported}'
        )

    unpaired = Counter((assertion.target, assertion.expected) for assertion in static) - Counter(
        (assertion.target, assertion.expected) for assertion in runtime
    )
    if unpaired:
        reported = '\n'.join(
            f'{ASSERTIONS[1]}({target}, {expected})' for target, expected in unpaired
        )
        pytest.fail(
            f'Every `{ASSERTIONS[0]}` needs a matching `{ASSERTIONS[1]}`, so that the '
            f'static and runtime halves pin the same type. Missing from {case.name}:\n{reported}'
        )
