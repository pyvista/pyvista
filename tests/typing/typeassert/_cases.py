"""Split a case file into the setup it needs and the cases it declares."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from pathlib import Path
    from types import CodeType

ASSERTION = 'assert_types'
SKIP_RUNTIME = 'SKIP_RUNTIME'


class CaseError(Exception):
    """Raised when a case file is not shaped the way the framework expects."""


class CaseSkipped(Exception):  # noqa: N818
    """Raised instead of running a case the file asks to skip at runtime."""


@dataclass(frozen=True)
class Case:
    """One `assert_types(expression, ExpectedType)` line."""

    path: Path
    lines: frozenset[int]
    expression: str
    expected: str
    code: CodeType

    @property
    def id(self) -> str:
        """Return the test id, which reads as the claim the case makes."""
        return f'{self.expression} -> {self.expected}'


@dataclass(frozen=True)
class CaseFile:
    """One case file: the cases it declares, and the setup they share."""

    path: Path
    cases: tuple[Case, ...]
    setup_lines: frozenset[int]
    setup_code: CodeType | None = None
    error: str | None = None

    @property
    def name(self) -> str:
        """Return the file name, used to scope test ids."""
        return self.path.name

    def setup_namespace(self) -> dict[str, Any]:
        """Execute this file's setup and return the namespace it produced."""
        namespace: dict[str, Any] = {'__name__': self.path.stem, '__file__': str(self.path)}
        # Running the case file's own code is the point of the framework.
        exec(self.setup_code, namespace)  # noqa: S102
        return namespace

    def run(self, case: Case) -> None:
        """Execute one case, and only it, against a fresh copy of the setup.

        Rebuilding the setup for every case is what keeps cases independent: one
        cannot observe another's state, and reordering the file changes nothing.
        """
        namespace = self.setup_namespace()
        reason = skip_reason(namespace, case)
        if reason is not None:
            raise CaseSkipped(reason)
        exec(case.code, namespace)  # noqa: S102

    def unknown_skips(self, namespace: dict[str, Any]) -> list[str]:
        """Return `SKIP_RUNTIME` keys that match no case, so a stale one is caught."""
        declared = namespace.get(SKIP_RUNTIME) or {}
        expressions = {case.expression for case in self.cases}
        return sorted(key for key in declared if key not in expressions)


def skip_reason(namespace: dict[str, Any], case: Case) -> str | None:
    """Return why `case` should not run, from the file's `SKIP_RUNTIME` mapping.

    A case file maps an expression to the reason running it would fail — a
    platform crash, an unavailable dependency. Mypy still checks the case, so
    only the runtime half is skipped. Building the mapping conditionally is
    ordinary Python, since it is read after the file's setup has run.
    """
    declared = namespace.get(SKIP_RUNTIME) or {}
    return declared.get(case.expression) or None


def _case_call(node: ast.AST) -> ast.Call | None:
    """Return the `assert_types` call a top-level statement makes, if it makes one."""
    if not isinstance(node, ast.Expr):
        return None
    call = node.value
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        return None
    return call if call.func.id == ASSERTION else None


def _reject_nested_assertions(tree: ast.Module, path: Path) -> None:
    """Reject `assert_types` calls that are not statements at module level.

    Such a call still type-checks, but it never becomes a case of its own, so it
    would be silently left out of the runtime half.
    """
    top_level = {id(call) for call in map(_case_call, tree.body) if call is not None}
    nested = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == ASSERTION
        and id(node) not in top_level
    ]
    if nested:
        lines = ', '.join(str(node.lineno) for node in nested)
        msg = (
            f'{path.name}: `{ASSERTION}` must be a statement at module level so that it '
            f'becomes a case of its own. Found one nested at line(s) {lines}.'
        )
        raise CaseError(msg)


def collect_case_file(path: Path) -> CaseFile:
    """Parse one case file into its cases and the setup they share.

    A file this cannot make sense of yields a `CaseFile` carrying the reason
    rather than raising, so a malformed file fails its own test instead of
    aborting collection for the session.
    """
    try:
        return _parse_case_file(path)
    except (CaseError, SyntaxError, OSError) as error:
        return CaseFile(path=path, cases=(), setup_lines=frozenset(), error=str(error))


def _parse_case_file(path: Path) -> CaseFile:
    """Parse one case file, raising `CaseError` if it is not shaped as expected."""
    source = path.read_text(encoding='utf-8')
    tree = ast.parse(source, filename=str(path))
    _reject_nested_assertions(tree, path)

    setup = [node for node in tree.body if _case_call(node) is None]
    setup_module = ast.Module(body=setup, type_ignores=[])
    setup_code = compile(ast.fix_missing_locations(setup_module), str(path), 'exec')

    cases = []
    for node in tree.body:
        call = _case_call(node)
        if call is None:
            continue
        if len(call.args) != 2:
            msg = f'{path.name}:{node.lineno}: `{ASSERTION}` takes an expression and a type.'
            raise CaseError(msg)
        module = ast.Module(body=[node], type_ignores=[])
        cases.append(
            Case(
                path=path,
                lines=frozenset(range(node.lineno, (node.end_lineno or node.lineno) + 1)),
                expression=ast.unparse(call.args[0]),
                expected=ast.unparse(call.args[1]),
                code=compile(ast.fix_missing_locations(module), str(path), 'exec'),
            )
        )

    case_lines = frozenset().union(*(case.lines for case in cases)) if cases else frozenset()
    return CaseFile(
        path=path,
        cases=tuple(cases),
        setup_lines=frozenset(range(1, len(source.splitlines()) + 1)) - case_lines,
        setup_code=setup_code,
    )


def collect_cases(directory: Path) -> list[CaseFile]:
    """Parse every case file in `directory`."""
    return [collect_case_file(path) for path in sorted(directory.glob('*.py'))]
