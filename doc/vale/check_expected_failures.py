"""Assert that Vale still flags the headings it is supposed to flag.

``doc/vale/headings.rst`` covers the other direction: it sits inside the paths
Vale scans, so every heading in it has to pass. Nothing there would notice the
rule going *slack*, which is the failure this repository actually had -- a bare
``a`` in the exceptions list matched inside every word containing the letter
and silently switched heading checks off for most of the documentation.

Run with ``make docstyle``, or directly::

    python3 doc/vale/check_expected_failures.py
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'tests' / 'vale' / 'headings_invalid.rst'
CONFIG = ROOT / 'doc' / '.vale.ini'
CHECK = 'Google.Headings'


def headings_in(path: Path) -> list[str]:
    """Return every RST section title in ``path``."""
    lines = path.read_text().splitlines()
    return [
        title.strip()
        for title, under in zip(lines, lines[1:])
        if title.strip()
        and re.fullmatch(r'[=\-~^"\'`#*+]{2,}', under.strip() or 'x')
        and len(under.strip()) >= len(title.strip())
    ]


def flagged_by_vale(path: Path) -> set[str]:
    """Return the headings in ``path`` that Vale reports for ``CHECK``."""
    out = subprocess.run(
        ['vale', '--config', str(CONFIG), '--output=JSON', str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        alert['Match']
        for alerts in json.loads(out.stdout or '{}').values()
        for alert in alerts
        if alert['Check'] == CHECK
    }


def main() -> int:
    if shutil.which('vale') is None:
        print('vale is not installed; skipping')
        return 0

    expected = headings_in(FIXTURE)[1:]  # the file's own title has to pass
    missed = [h for h in expected if h not in flagged_by_vale(FIXTURE)]
    if missed:
        print(f'{CHECK} no longer flags these headings:')
        for heading in missed:
            print(f'  {heading}')
        print(f'\nSee {FIXTURE.relative_to(ROOT)}')
        return 1

    print(f'{CHECK}: all {len(expected)} expected failures still caught')
    return 0


if __name__ == '__main__':
    sys.exit(main())
