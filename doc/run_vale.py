"""Run Vale the way CI runs it.

The set of paths Vale checks is defined here and nowhere else: `make docstyle`,
`CONTRIBUTING.rst` and the `Style and Docstring Check` workflow all reach it
through this file, so adding a path is a one-line change. The workflow reads
the list with ``--print-files`` rather than repeating it, because it runs Vale
through `vale-action` to get inline annotations on the pull request.

Usage::

    python3 doc/run_vale.py                # extract, lint, check the fixtures
    python3 doc/run_vale.py --print-files  # the path list, as JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'doc' / '.vale.ini'

# Gallery examples and docstrings are extracted to .rst first, so Vale sees them
# with numpydoc structure resolved; see `doc/extract_rst_from_py_for_vale.py`.
# The last two entries are fixtures that must stay valid, and so have to be
# named explicitly. Their `_invalid` counterparts are deliberately absent: those
# cases must fail, which `check_expected_failures.py` asserts.
PATHS = [
    'doc',
    'pyvista',
    'examples',
    'CONTRIBUTING.rst',
    '.vale/examples',
    '.vale/pyvista',
    'tests/doc/vale/headings.rst',
    'tests/doc/vale/repetition.rst',
]

EXTRACT = [
    ['examples', '.vale/examples'],
    ['pyvista', '.vale/pyvista', '--mode', 'docstrings'],
]


def run(command: list[str]) -> int:
    """Echo and run ``command`` from the repository root."""
    print('+', ' '.join(command))
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def main() -> int:
    """Extract, lint, and check the fixtures; return an exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--print-files',
        action='store_true',
        help='print the paths Vale checks as a JSON array, then exit',
    )
    args = parser.parse_args()

    if args.print_files:
        print(json.dumps(PATHS))
        return 0

    if shutil.which('vale') is None:
        print("vale is not installed: pip install vale 'docutils<0.22' 'sphinx-gallery<0.22.0'")
        return 1

    for extract in EXTRACT:
        code = run([sys.executable, 'doc/extract_rst_from_py_for_vale.py', *extract])
        if code:
            return code

    code = run(['vale', '--config', str(CONFIG.relative_to(ROOT)), *PATHS])
    if code:
        return code

    return run([sys.executable, 'tests/doc/vale/check_expected_failures.py'])


if __name__ == '__main__':
    sys.exit(main())
