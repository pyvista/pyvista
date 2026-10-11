"""Flag double spaces between sentences, in place of Vale's ``Google.Spacing`` check.

Vale masks inline code and roles and skips some paragraphs outright, so it misses many of
these. This checks ``.rst`` files, docstrings and gallery text directly, in pre-commit.
``Google.Spacing`` checks only for a missing space.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
import re
import sys

# A sentence end that is not an ellipsis, then two or more spaces, then the next sentence.
PATTERN = re.compile(r"""[^\s.][.?!]["')]? {2,}(?!"{3})[A-Z0-9`('":*|[]""")
ENUMERATOR = re.compile(r'^\s*(?:\d+|[A-Za-z]|[ivxlcdm]+|[IVXLCDM]+|#)\.(?= )')
DOCTEST = re.compile(r'^\s*(?:>>>|\.\.\.)(?: |$)')
CODE_DIRECTIVE = re.compile(
    r'^\.\. (?:code-block|code|sourcecode|parsed-literal|literalinclude|highlight|math'
    r'|doctest|testcode|testsetup|testcleanup|testoutput|ipython|jupyter-execute'
    r'|pyvista-plot|plot|command-output|raw|graphviz)::',
    re.IGNORECASE,
)
COMMENT = re.compile(r'^\.\.(?: (?![\w-]+::|_|\[|\|)|$)')
TABLE_BORDER = re.compile(r'^=+(?: +=+)+$')
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
    """Return the lines of ``path`` that hold prose, as RST text, with their line numbers."""
    source = path.read_text(encoding='utf-8')
    lines = source.splitlines()
    gallery: set[int] = set()
    if path.suffix == '.rst':
        numbers = set(range(1, len(lines) + 1))
    elif Path(os.path.relpath(path)).parts[0] == 'examples':
        gallery = _gallery_text_lines(source)
        numbers = _docstring_lines(source, module_only=True) | gallery
    else:
        numbers = _docstring_lines(source)
    return [
        (n, re.sub(r'^# ?', '', lines[n - 1]) if n in gallery else lines[n - 1])
        for n in sorted(numbers)
        if n <= len(lines)
    ]


def violations(path: Path) -> list[tuple[int, str]]:
    """Return the lines of ``path`` with a double space between sentences."""
    found = []
    lines = prose_lines(path)
    previous = None
    literal_indent = None  # inside a literal block deeper than this indent
    item_indent = None  # inside a list item at this indent
    in_doctest = in_table = False
    paragraph_start = True
    for index, (number, text) in enumerate(lines):
        if previous is None or number != previous + 1:
            literal_indent, item_indent, in_doctest, paragraph_start = None, None, False, True
            in_table = False
        previous = number
        stripped = text.strip()
        indent = len(text) - len(text.lstrip())
        if not stripped:
            in_doctest, paragraph_start = False, True
            continue
        if literal_indent is not None:
            if indent > literal_indent:
                continue
            literal_indent = None
        if TABLE_BORDER.match(stripped):
            following = lines[index + 1] if index + 1 < len(lines) else None
            ends = following is None or following[0] != number + 1 or not following[1].strip()
            in_table = not (in_table and ends)
            continue
        if in_table:
            continue
        if COMMENT.match(stripped):
            literal_indent = indent
            continue
        if in_doctest or DOCTEST.match(text):
            in_doctest = True
            continue
        if item_indent is not None and indent <= item_indent and not ENUMERATOR.match(text):
            item_indent = None
        body = text
        if (match := ENUMERATOR.match(text)) and (paragraph_start or item_indent is not None):
            body, item_indent = text[match.end() :], indent
        if PATTERN.search(body):
            found.append((number, stripped))
        if CODE_DIRECTIVE.match(stripped) or (
            stripped.endswith('::') and not stripped.startswith('..')
        ):
            literal_indent = indent
        paragraph_start = False
    return found


def main(argv: list[str]) -> int:
    """Check each file in argv, printing every double space between sentences."""
    exit_code = 0
    for filename in argv:
        for number, text in violations(Path(filename)):
            print(f'{filename}:{number}: {text}')  # noqa: T201
            exit_code = 1
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
