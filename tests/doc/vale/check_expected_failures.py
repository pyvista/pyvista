"""Assert that Vale still flags what it is supposed to flag.

``headings.rst`` and ``repetition.rst`` cover the other direction: they sit
inside the paths Vale scans, so everything in them has to pass. Nothing there
would notice a rule going *slack*, which is the failure this repository
actually had -- a bare ``a`` in ``Google.Headings``' exceptions list matched
inside every word containing the letter and silently switched heading checks
off for most of the documentation.

``PyVista.Repetition`` has no exceptions list, so its ``tokens`` pattern is the
only thing standing between a real doubled word and silence.

``rules_invalid.rst`` covers the other rules ``doc/.vale.ini`` raises to ``error``,
and every such rule needs a case there.

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
RULES_FIXTURE = HERE / 'rules_invalid.rst'
RULES_SECTION = '[*.{md,rst}]'


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


def alerts_for(path: Path, check: str | None = None) -> list[dict]:
    """Return the alerts Vale reports in ``path``, only for ``check`` if given."""
    out = subprocess.run(
        ['vale', '--config', str(CONFIG), '--output=JSON', str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode == 2:
        msg = f'vale failed on {path.name}:\n{out.stderr or out.stdout}'
        raise RuntimeError(msg)
    return [
        alert
        for alerts in json.loads(out.stdout or '{}').values()
        for alert in alerts
        if check is None or alert['Check'] == check
    ]


def expectations_in(path: Path) -> dict[int, tuple[str, str]]:
    """Map the line after each ``.. expect: <Check>`` comment to its check and text."""
    lines = path.read_text().splitlines()
    expected = {}
    for index, line in enumerate(lines):
        if match := re.fullmatch(r'\.\. expect: (\S+)', line.strip()):
            target = next((i for i in range(index + 1, len(lines)) if lines[i].strip()), None)
            if target is None:
                msg = f'{path.name}:{index + 1}: expect comment has no case after it'
                raise ValueError(msg)
            expected[target + 1] = (match[1], lines[target].strip())
    return expected


def error_rules(config: Path) -> set[str]:
    """Return the rules ``config`` raises to ``error`` for ``.md`` and ``.rst`` files."""
    rules = set()
    in_section = False
    for line in config.read_text().splitlines():
        if line.startswith('['):
            in_section = line.strip() == RULES_SECTION
        elif in_section and (match := re.fullmatch(r'([\w.]+) = error', line.strip())):
            rules.add(match[1])
    return rules


def no_cases(fixture: Path) -> int:
    """Report a fixture with no cases in it, which would otherwise pass vacuously."""
    print(f'No expected failures found in {fixture.relative_to(ROOT)}')
    return 1


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
    if not expected:
        return no_cases(HEADINGS_FIXTURE)
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
    if not expected:
        return no_cases(REPETITION_FIXTURE)
    missed = [text for number, text in expected.items() if number not in flagged]
    if missed:
        print(f'{REPETITION_CHECK} no longer flags these:')
        for text in missed:
            print(f'  {text}')
        print(f'\nSee {REPETITION_FIXTURE.relative_to(ROOT)}')
        return 1

    print(f'{REPETITION_CHECK}: all {len(expected)} expected failures still caught')
    return 0


def check_rules() -> int:
    """Report any case in the rules fixture its rule no longer catches."""
    flagged = {(alert['Line'], alert['Check']) for alert in alerts_for(RULES_FIXTURE)}
    expected = expectations_in(RULES_FIXTURE)
    if not expected:
        return no_cases(RULES_FIXTURE)
    uncovered = sorted(
        error_rules(CONFIG) - {HEADINGS_CHECK} - {check for check, _ in expected.values()}
    )
    if uncovered:
        print(f'These rules are errors in {CONFIG.name} but have no case:')
        for rule in uncovered:
            print(f'  {rule}')
        print(f'\nAdd one to {RULES_FIXTURE.relative_to(ROOT)}')
        return 1
    missed = [
        f'{check}: {text}'
        for number, (check, text) in expected.items()
        if (number, check) not in flagged
    ]
    if missed:
        print('These rules no longer flag their case:')
        for case in missed:
            print(f'  {case}')
        print(f'\nSee {RULES_FIXTURE.relative_to(ROOT)}')
        return 1

    print(f'Error-level rules: all {len(expected)} expected failures still caught')
    return 0


def main() -> int:
    """Report any expected failure a rule no longer catches."""
    if shutil.which('vale') is None:
        print('vale is not installed; skipping')
        return 0

    # All run, so one slack rule does not hide another's report.
    return max(check_headings(), check_repetition(), check_rules())


if __name__ == '__main__':
    sys.exit(main())
