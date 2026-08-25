"""Extract the RST prose from Sphinx-Gallery example scripts for Vale.

Sphinx-Gallery examples embed RST in a leading docstring and in ``# %%``
comment cells, which Vale cannot parse for structure (headings, etc.) since
they are just Python comments on disk. This mirrors ``examples/`` into a
scratch directory of plain ``.rst`` files -- text blocks only, no code --
using Sphinx-Gallery's own (non-executing) source parser, so Vale can lint
them like any other doc.

Blank lines are inserted so each block keeps its original line number,
making Vale's file:line output point back to roughly the right spot in the
source ``.py`` file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sphinx_gallery.py_source_parser import split_code_and_text_blocks


def convert_file(py_path: Path, rst_path: Path) -> None:
    """Write the RST text blocks of one example script to ``rst_path``."""
    _, blocks = split_code_and_text_blocks(str(py_path))

    lines: list[str] = []
    for label, content, lineno in blocks:
        if label != 'text':
            continue
        while len(lines) < lineno - 1:
            lines.append('')
        lines.extend(content.splitlines())

    rst_path.parent.mkdir(parents=True, exist_ok=True)
    rst_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    """Convert every example script under ``src`` into RST files under ``dest``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('src', type=Path, help='Directory of example scripts, e.g. examples/')
    parser.add_argument('dest', type=Path, help='Output directory for the generated .rst files')
    args = parser.parse_args()

    for py_path in sorted(args.src.rglob('*.py')):
        rst_path = args.dest / py_path.relative_to(args.src).with_suffix('.rst')
        convert_file(py_path, rst_path)


if __name__ == '__main__':
    main()
