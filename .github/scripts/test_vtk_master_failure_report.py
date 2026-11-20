"""Test the VTK master nightly's failure summary and report scripts on fabricated logs.

Run with ``pytest -c /dev/null --rootdir=. .github/scripts/test_vtk_master_failure_report.py``; the
``test-failure-report`` job of ``vtk-master-wheels.yml`` does, before any VTK build.
"""

from __future__ import annotations

from pathlib import Path
import random
import subprocess

import pytest

SCRIPTS = Path(__file__).parent
SUMMARY = SCRIPTS / 'vtk-master-failure-summary.sh'
REPORT = SCRIPTS / 'vtk-master-failure-report.sh'

ACTOR_REPR = (
    'Actor (0x7fd394bf7880)\\n  Center: (37.2, -0.5, 0.0)\\n  Pickable: True\\n'
    "Property (0x7fd394bf7990)\\n  Ambient: 0.0\\n  Color: Color(name='tomato')\\n" * 40
)
SHORT = [
    f'FAILED tests/plotting/test_plotting.py::test_case[{i}] - ImportError: x' for i in range(100)
]
LONG = [
    f'FAILED tests/plotting/test_widgets.py::test_affine_widget[{style}] - '
    f'assert None in ([{ACTOR_REPR}])'
    for style in ('trackball_style', 'zoom_style', 'terrain_style')
]


def tox_log(failed: list[str], *, passed: int = 3384, eol: str = '\n', color: bool = False) -> str:
    """Return a tox log ending in pytest's short summary and final line."""
    esc = '\x1b[31m' if color else ''
    reset = '\x1b[0m' if color else ''
    lines = [
        'py3.14-plotting-vtk_local: commands[0]> pytest',
        'tests/plotting/test_plotting.py::test_pass PASSED',
        f'{esc}=============== short test summary info ==============={reset}',
        *failed,
        f'{esc}= {len(failed)} failed, {passed} passed, 24 skipped in 392.95s (0:06:32) ={reset}',
        'py3.14-plotting-vtk_local: FAIL code 1 (401.12=setup[8.17]+cmd[392.95] seconds)',
        'evaluation failed :( (401.30 seconds)',
    ]
    return eol.join(lines) + eol


def shuffled(failed: list[str], seed: int) -> list[str]:
    """Return the lines in another order with other object addresses."""
    rng = random.Random(seed)
    lines = [line.replace('0x7fd394bf7880', f'0x{rng.getrandbits(44):x}') for line in failed]
    rng.shuffle(lines)
    return lines


@pytest.fixture
def summarize(tmp_path: Path):
    """Run the summary script on tox logs and return (summary, n_failures)."""
    counter = iter(range(10_000))

    def run(*logs: str) -> tuple[str, int]:
        out = tmp_path / f'out{next(counter)}'
        out.mkdir()
        paths = []
        for i, log in enumerate(logs):
            path = out / f'log{i}.log'
            path.write_bytes(log.encode())
            paths.append(str(path))
        subprocess.run(['bash', str(SUMMARY), str(out), *paths], check=True)
        summary = (out / 'failure_summary.txt').read_text()
        return summary, int((out / 'n_failures.txt').read_text())

    return run


@pytest.fixture
def report():
    """Run a report helper on a summary and return its output."""

    def run(command: str, summary: str, *args: str) -> str:
        return subprocess.run(
            ['bash', str(REPORT), command, *args],
            input=summary,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    return run


def test_every_failure_survives_long_lines(summarize):
    summary, n_failures = summarize(tox_log(SHORT + LONG))
    names = [line.split(' - ')[0] for line in summary.splitlines()]
    assert names == sorted(line.split(' - ')[0] for line in SHORT + LONG)
    assert n_failures == len(SHORT + LONG)
    assert max(map(len, summary.splitlines())) == 203


def test_count_adds_errors_and_both_logs(summarize):
    core = tox_log(['ERROR tests/core/test_a.py::test_a - boom'] * 2)
    core = core.replace('= 2 failed,', '= 3 failed, 2 errors,')
    plotting = tox_log(SHORT[:5])
    summary, n_failures = summarize(core, plotting)
    assert n_failures == 10
    assert summary.count('\n') == 7


def test_log_without_summary_keeps_the_tail(summarize):
    lines = [f'line {i}' for i in range(100)] + ['py3.14-core-vtk_local: FAIL code 1']
    summary, n_failures = summarize('\n'.join(lines) + '\n')
    assert n_failures == 0
    assert summary.splitlines() == lines[-40:]


def test_windows_line_endings_and_color(summarize):
    summary, n_failures = summarize(tox_log(SHORT[:3], eol='\r\n', color=True))
    assert '\r' not in summary
    assert '\x1b' not in summary
    assert n_failures == 3
    assert summary.splitlines() == SHORT[:3]


def test_summary_fits_a_comment(summarize):
    many = [f'FAILED tests/test_x.py::test_{i} - ' + 'x' * 300 for i in range(500)]
    summary, n_failures = summarize(tox_log(many))
    assert len(summary.encode()) <= 16000
    assert n_failures == 500
    assert summary.endswith('...\n')


def test_hash_ignores_order_addresses_and_timing(summarize, report):
    first, _ = summarize(tox_log(SHORT + LONG))
    second, _ = summarize(tox_log(shuffled(SHORT + LONG, seed=1)).replace('392.95s', '401.00s'))
    assert report('hash', first) == report('hash', second)


def test_hash_changes_with_the_failing_set(summarize, report):
    base, _ = summarize(tox_log(SHORT))
    fewer, _ = summarize(tox_log(SHORT[:-1]))
    renamed, _ = summarize(tox_log([*SHORT[:-1], SHORT[-1].replace('[99]', '[100]')]))
    assert len({report('hash', s) for s in (base, fewer, renamed)}) == 3


@pytest.mark.parametrize(
    ('count', 'n_lines', 'expected'),
    [
        (93, 92, '93 failures, first 92 shown'),
        (93, 93, '93 failures'),
        (0, 5, '5 failures'),
        ('', 1, '1 failure'),
        (0, 0, 'failed before pytest reported (end of log)'),
    ],
)
def test_summary_label(report, count, n_lines, expected):
    summary = '\n'.join(SHORT[:n_lines]) if n_lines else 'tox: exit 1'
    assert report('label', summary, str(count)) == expected
