"""Enforce Sphinx-Gallery example docstring structure.

Ruff's D400/D415 (first line ends in punctuation) can't be enabled for
``examples/`` because Sphinx-Gallery requires the docstring to start with a
``.. _label:`` target followed by a title/underline pair, which those rules
would force us to break. This hook instead enforces the shape directly: a
ref label, a title with a matching underline, and a summary right after the
title that fits on a single line and ends in punctuation, mirroring D400.
"""

from __future__ import annotations

import ast
from pathlib import Path
import sys

UNDERLINE_CHARS = set('~-=^"#*+.')


def _title_error(paragraph: str) -> str | None:
    """Check that the title paragraph is a title line plus a matching underline."""
    title_lines = paragraph.strip('\n').split('\n')
    if len(title_lines) != 2:
        return 'summary must be its own paragraph, separated from the title by a blank line'
    title, underline = title_lines
    underline_ok = (
        len(set(underline)) == 1
        and underline[0] in UNDERLINE_CHARS
        and len(underline) >= len(title)
    )
    return (
        None if underline_ok else f'title underline {underline!r} does not match title {title!r}'
    )


def _summary_error(paragraph: str) -> str | None:
    """Check that the summary paragraph is a single line ending in punctuation."""
    summary = paragraph.strip('\n')
    if '\n' in summary:
        return f'summary must fit on a single line, not wrap: {summary!r}'
    if summary.startswith('.. '):
        return 'summary paragraph is a directive, not a summary sentence'
    return (
        None
        if summary and summary[-1] in '.?!'
        else f'summary line does not end in punctuation: {summary!r}'
    )


def find_docstring_error(source: str) -> str | None:
    """Return a description of the first docstring structure violation, if any."""
    doc = ast.get_docstring(ast.parse(source), clean=False)
    if doc is None:
        return 'module is missing a docstring'

    paragraphs = doc.strip('\n').split('\n\n')
    if not paragraphs[0].lstrip().startswith('.. _'):
        return 'docstring must start with a ``.. _label:`` target'
    if len(paragraphs) < 3:
        return 'docstring is missing a title or a summary paragraph after it'

    return _title_error(paragraphs[1]) or _summary_error(paragraphs[2])


def main(argv: list[str]) -> int:
    """Check each file in argv, printing and counting violations."""
    exit_code = 0
    for filename in argv:
        error = find_docstring_error(Path(filename).read_text(encoding='utf-8'))
        if error is not None:
            print(f'{filename}: {error}')  # noqa: T201
            exit_code = 1
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
