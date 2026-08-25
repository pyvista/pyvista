"""Convert Python sources to .rst files so Vale can lint their prose.

Two modes: ``gallery`` extracts the ``# %%`` cell text of Sphinx-Gallery
example scripts; ``docstrings`` extracts every module/class/function
docstring via ``ast``. Both pad the rest of each file with blank lines so
line numbers still point back to the source.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

from sphinx_gallery.py_source_parser import split_code_and_text_blocks


def convert_gallery_file(py_path: Path, rst_path: Path) -> None:
    """Write the RST text blocks of one example script to ``rst_path``."""
    _, blocks = split_code_and_text_blocks(str(py_path))

    lines: list[str] = []
    for label, content, lineno in blocks:
        if label != 'text':
            continue
        while len(lines) < lineno - 1:  # pad so line numbers still match the source
            lines.append('')
        lines.extend(content.splitlines())

    rst_path.parent.mkdir(parents=True, exist_ok=True)
    rst_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


# numpydoc sections shaped like "name : type\n    description" -- only the
# description is prose; the signature line isn't real RST.
STRUCTURED_SECTIONS = {
    'Parameters',
    'Other Parameters',
    'Attributes',
    'Methods',
    'Returns',
    'Yields',
    'Raises',
    'Warns',
    'Receives',
}
# Sections that are code or bare references, not prose.
SKIP_SECTIONS = {'Examples', 'See Also'}


def _is_section_header(header: str, underline: str) -> bool:
    """Match numpydoc's own rule for what counts as a section header."""
    return len(underline) >= 3 and underline.startswith(('-' * len(header), '=' * len(header)))


def filter_numpydoc(lines: list[str]) -> list[str]:
    """Blank out numpydoc signature lines and skip-section bodies, keep the rest."""
    out = list(lines)
    mode = 'prose'
    i = 0
    while i < len(lines):
        line = lines[i]
        if line and not line[0].isspace() and i + 1 < len(lines):
            header = line.strip()
            if _is_section_header(header, lines[i + 1].strip()):
                if header in STRUCTURED_SECTIONS:
                    mode = 'structured'
                elif header in SKIP_SECTIONS:
                    mode = 'skip'
                else:
                    mode = 'prose'
                i += 2
                continue

        if mode == 'skip':
            out[i] = ''
        elif mode == 'structured' and line.strip() and not line[0].isspace():
            out[i] = ''  # "name : type" signature line
        i += 1
    return out


def convert_docstring_file(py_path: Path, rst_path: Path) -> None:
    """Write every docstring in one module to ``rst_path``, in source order."""
    try:
        tree = ast.parse(py_path.read_text(encoding='utf-8'))
    except SyntaxError:
        return

    nodes = [
        tree,
        *(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        ),
    ]
    docstrings = []
    for node in nodes:
        doc = ast.get_docstring(node, clean=True)
        if doc is not None:
            docstrings.append((node.body[0].lineno, doc))
    docstrings.sort(key=lambda item: item[0])

    lines: list[str] = []
    for lineno, doc in docstrings:
        while len(lines) < lineno - 1:
            lines.append('')
        lines.extend(filter_numpydoc(doc.splitlines()))

    rst_path.parent.mkdir(parents=True, exist_ok=True)
    rst_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    """Convert every .py file under ``src`` into RST files under ``dest``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('src', type=Path, help='Directory of .py files, e.g. examples/')
    parser.add_argument('dest', type=Path, help='Output directory for the generated .rst files')
    parser.add_argument('--mode', choices=['gallery', 'docstrings'], default='gallery')
    args = parser.parse_args()

    convert_file = convert_gallery_file if args.mode == 'gallery' else convert_docstring_file
    for py_path in sorted(args.src.rglob('*.py')):
        rst_path = args.dest / py_path.relative_to(args.src).with_suffix('.rst')
        convert_file(py_path, rst_path)


if __name__ == '__main__':
    main()
