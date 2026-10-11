"""Flag two or more spaces between sentences in ``.rst`` files, docstrings and gallery text."""

from __future__ import annotations

import ast
from pathlib import Path
import re
import sys

# Same token as doc/styles/Google/Spacing.yml.
PATTERN = re.compile(r'[^\s.][.?!] {2,}[A-Z]')
ENUMERATOR = re.compile(r'^[\s#]*(?:\d+|[A-Za-z]|#)\.(?= )')
DOCTEST = re.compile(r'^\s*(?:>>>|\.\.\.)(?: |$)')
GALLERY_CELL = re.compile(r'^# ?(?:%%|#{20,})')


def _docstring_lines(source: str, *, module_only: bool = False) -> set[int]:
    """Return the line numbers that belong to a module, class or function docstring."""
    tree = ast.parse(source)
    lines = set()
    for node in [tree] if module_only else ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                lines.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    return lines


def _gallery_text_lines(source: str) -> set[int]:
    """Return the line numbers of the comment lines in Sphinx-Gallery text blocks."""
    lines = set()
    in_text = False
    for number, line in enumerate(source.splitlines(), start=1):
        if GALLERY_CELL.match(line):
            in_text = True
        elif in_text and line.startswith('#'):
            lines.add(number)
        else:
            in_text = False
    return lines


def prose_lines(path: Path) -> list[tuple[int, str]]:
    """Return the prose lines of ``path`` with their line numbers."""
    source = path.read_text(encoding='utf-8')
    lines = source.splitlines()
    if path.suffix == '.rst':
        numbers = set(range(1, len(lines) + 1))
    elif path.parts[0] == 'examples':
        numbers = _docstring_lines(source, module_only=True) | _gallery_text_lines(source)
    else:
        numbers = _docstring_lines(source)
    return [(n, lines[n - 1]) for n in sorted(numbers) if n <= len(lines)]


def main(argv: list[str]) -> int:
    """Check each file in argv, printing every double space between sentences."""
    exit_code = 0
    for filename in argv:
        for number, line in prose_lines(Path(filename)):
            if not DOCTEST.match(line) and PATTERN.search(ENUMERATOR.sub('', line)):
                print(f'{filename}:{number}: {line.strip()}')  # noqa: T201
                exit_code = 1
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
