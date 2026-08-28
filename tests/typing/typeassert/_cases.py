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


class CaseError(Exception):
    """Raised when a case file is not shaped the way the framework expects."""


@dataclass(frozen=True)
class Case:
    """One `assert_types(expression, ExpectedType)` line, runnable on its own."""

    path: Path
    lines: frozenset[int]
    expression: str
    expected: str
    code: CodeType

    @property
    def id(self) -> str:
        """Return the test id, which reads as the claim the case makes."""
        return f'{self.expression} -> {self.expected}'

    def run(self) -> None:
        """Execute this case's line, and only it, in a namespace of its own."""
        namespace: dict[str, Any] = {'__name__': self.path.stem, '__file__': str(self.path)}
        # Running the case file's own code is the point of the framework.
        exec(self.code, namespace)  # noqa: S102


@dataclass(frozen=True)
class CaseFile:
    """One case file: the lines that assert something, and the lines that do not."""

    path: Path
    cases: tuple[Case, ...]
    setup_lines: frozenset[int]
    error: str | None = None

    @property
    def name(self) -> str:
        """Return the file name, used to scope test ids."""
        return self.path.name


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

    Each case compiles to the file's setup statements followed by that one case,
    so running it rebuilds everything it depends on from scratch. Cases cannot
    reach each other's state, and none of this depends on line order.

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
    cases = []
    for node in tree.body:
        call = _case_call(node)
        if call is None:
            continue
        if len(call.args) != 2:
            msg = f'{path.name}:{node.lineno}: `{ASSERTION}` takes an expression and a type.'
            raise CaseError(msg)
        module = ast.Module(body=[*setup, node], type_ignores=[])
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
    setup_lines = frozenset(range(1, len(source.splitlines()) + 1)) - case_lines
    return CaseFile(path=path, cases=tuple(cases), setup_lines=setup_lines)


def collect_cases(directory: Path) -> list[CaseFile]:
    """Parse every case file in `directory`."""
    return [collect_case_file(path) for path in sorted(directory.glob('*.py'))]
