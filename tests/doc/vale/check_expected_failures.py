"""Assert that Vale still flags what it is supposed to flag.

``headings.rst`` and ``repetition.rst`` cover the other direction: they sit
inside the paths Vale scans, so everything in them has to pass. Nothing there
would notice a rule going *slack*, which is the failure this repository
actually had -- a bare ``a`` in ``Google.Headings``' exceptions list matched
inside every word containing the letter and silently switched heading checks
off for most of the documentation.

``PyVista.Repetition`` is the same shape of risk from the other direction: it
carried an exceptions list of type names purely because it ran over raw ``.py``
files, where numpydoc type lines and ``See Also`` entries reach it as prose.
The list is gone, so its ``tokens`` pattern is now the only thing standing
between a real doubled word and silence.

Run with ``make docstyle``, or directly::

    python3 tests/doc/vale/check_expected_failures.py
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CONFIG = ROOT / 'doc' / '.vale.ini'
HEADINGS_FIXTURE = HERE / 'headings_invalid.rst'
HEADINGS_CHECK = 'Google.Headings'
REPETITION_FIXTURE = HERE / 'repetition_invalid.rst'
REPETITION_CHECK = 'PyVista.Repetition'


def headings_in(path: Path) -> list[str]:
    """Return every RST section title in ``path``."""
    lines = path.read_text().splitlines()
    return [
        title.strip()
        for title, under in itertools.pairwise(lines)
        if title.strip()
        and re.fullmatch(r'[=\-~^"\'`#*+]{2,}', under.strip() or 'x')
        and len(under.strip()) >= len(title.strip())
    ]


def alerts_for(path: Path, check: str) -> list[dict]:
    """Return the alerts Vale reports in ``path`` for ``check``."""
    out = subprocess.run(
        ['vale', '--config', str(CONFIG), '--output=JSON', str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    return [
        alert
        for alerts in json.loads(out.stdout or '{}').values()
        for alert in alerts
        if alert['Check'] == check
    ]


def bullets_in(path: Path) -> dict[int, str]:
    """Return the ``- `` bullet lines in ``path``, keyed by line number."""
    return {
        number: line.strip()[2:]
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if line.startswith('- ')
    }


def check_headings() -> int:
    """Report any heading the rule no longer catches."""
    flagged = {alert['Match'] for alert in alerts_for(HEADINGS_FIXTURE, HEADINGS_CHECK)}
    expected = headings_in(HEADINGS_FIXTURE)[1:]  # the file's own title has to pass
    missed = [heading for heading in expected if heading not in flagged]
    if missed:
        print(f'{HEADINGS_CHECK} no longer flags these headings:')
        for heading in missed:
            print(f'  {heading}')
        print(f'\nSee {HEADINGS_FIXTURE.relative_to(ROOT)}')
        return 1

    print(f'{HEADINGS_CHECK}: all {len(expected)} expected failures still caught')
    return 0


def check_repetition() -> int:
    """Report any doubled word the rule no longer catches."""
    flagged = {alert['Line'] for alert in alerts_for(REPETITION_FIXTURE, REPETITION_CHECK)}
    expected = bullets_in(REPETITION_FIXTURE)
    missed = [text for number, text in expected.items() if number not in flagged]
    if missed:
        print(f'{REPETITION_CHECK} no longer flags these:')
        for text in missed:
            print(f'  {text}')
        print(f'\nSee {REPETITION_FIXTURE.relative_to(ROOT)}')
        return 1

    print(f'{REPETITION_CHECK}: all {len(expected)} expected failures still caught')
    return 0


def main() -> int:
    """Report any expected failure a rule no longer catches."""
    if shutil.which('vale') is None:
        print('vale is not installed; skipping')
        return 0

    # Both run, so one slack rule does not hide the other's report.
    return max(check_headings(), check_repetition())


if __name__ == '__main__':
    sys.exit(main())
