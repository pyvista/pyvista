"""Run Vale the way CI runs it.

The set of paths Vale checks is defined here and nowhere else: `make docstyle`,
`CONTRIBUTING.rst` and the `Style and Docstring Check` workflow all reach it
through this file, so adding a path is a one-line change.

Usage::

    python3 doc/run_vale.py             # extract, lint, check the fixtures
    python3 doc/run_vale.py --annotate  # the same, with GitHub annotations
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

# An extracted file mirrors its source line for line, so an annotation can point
# at the `.py` the reader has to edit rather than at the generated `.rst`.
SOURCES = {'.vale/examples': 'examples', '.vale/pyvista': 'pyvista'}

# GitHub has no `suggestion` level.
LEVELS = {'error': 'error', 'warning': 'warning', 'suggestion': 'notice'}

# A suggestion is advisory; anything above it fails the run. `MinAlertLevel` in
# doc/.vale.ini decides which of them Vale reports in the first place.
FAILING = {'warning', 'error'}


def run(command: list[str]) -> int:
    """Echo and run ``command`` from the repository root."""
    print('+', ' '.join(command))
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def escape(value: str, *, is_property: bool = False) -> str:
    """Escape ``value`` for a GitHub workflow command."""
    escaped = value.replace('%', '%25').replace('\r', '%0D').replace('\n', '%0A')
    if is_property:
        escaped = escaped.replace(':', '%3A').replace(',', '%2C')
    return escaped


def source_of(path: str) -> str:
    """Return the file an alert's reader has to edit, given the file Vale read."""
    for extracted, source in SOURCES.items():
        if path.startswith(extracted + '/'):
            return source + path[len(extracted) : -len('.rst')] + '.py'
    return path


def annotate(alerts: dict[str, list[dict]]) -> None:
    """Print one GitHub annotation per alert."""
    for path, file_alerts in sorted(alerts.items()):
        source = escape(source_of(path), is_property=True)
        for alert in file_alerts:
            level = LEVELS.get(alert['Severity'], 'error')
            start, end = alert['Span']
            title = escape(f'Vale: {alert["Check"]}', is_property=True)
            print(
                f'::{level} file={source},line={alert["Line"]},'
                f'col={start},endColumn={end},title={title}'
                f'::{escape(alert["Message"])}'
            )


def lint(*, annotations: bool) -> int:
    """Run Vale over ``PATHS``; return an exit status."""
    command = ['vale', '--config', str(CONFIG.relative_to(ROOT))]
    if not annotations:
        return run([*command, *PATHS])

    command += ['--output=JSON', *PATHS]
    print('+', ' '.join(command))
    vale = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    print(vale.stderr, end='')
    if vale.returncode > 1:  # Vale itself failed, and reported that instead of alerts
        print(vale.stdout, end='')
        return vale.returncode

    alerts = json.loads(vale.stdout or '{}')
    annotate(alerts)
    total = sum(len(file_alerts) for file_alerts in alerts.values())
    failing = [
        alert
        for file_alerts in alerts.values()
        for alert in file_alerts
        if alert['Severity'] in FAILING
    ]
    print(f'{total} alert(s) in {len(alerts)} file(s), {len(failing)} of them failing')
    return 1 if failing else 0


def main() -> int:
    """Extract, lint, and check the fixtures; return an exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--annotate',
        action='store_true',
        help='report each alert as a GitHub Actions annotation',
    )
    args = parser.parse_args()

    if shutil.which('vale') is None:
        print("vale is not installed: pip install vale 'docutils<0.22' 'sphinx-gallery<0.22.0'")
        return 1

    for extract in EXTRACT:
        code = run([sys.executable, 'doc/extract_rst_from_py_for_vale.py', *extract])
        if code:
            return code

    code = lint(annotations=args.annotate)
    if code:
        return code

    return run([sys.executable, 'tests/doc/vale/check_expected_failures.py'])


if __name__ == '__main__':
    sys.exit(main())
