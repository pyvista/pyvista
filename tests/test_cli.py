from __future__ import annotations

import contextlib
import io
import subprocess
import sys

import pytest

import pyvista as pv
from pyvista.__main__ import main


def pop_second_line(text: str) -> str:
    """Return text with the second line removed."""
    lines = text.splitlines()
    if len(lines) > 1:
        lines.pop(1)
    return '\n'.join(lines)


def _run_cli(argv: list[str] | None = None, *, as_script: bool = False):
    args = []
    if not as_script:
        args.extend([sys.executable, '-m'])
    args.append('pyvista')
    if argv:
        args.extend(argv)
    return subprocess.run(
        args,
        check=False,
        capture_output=True,
        encoding='utf-8',
    )


@pytest.mark.parametrize('args', [None, []])
def test_no_input(args):
    result = _run_cli(args)
    assert result.returncode == 1
    stdout = result.stdout
    assert stdout.startswith('usage: pyvista [-h] [--version] {report} ...')
    assert 'positional arguments:' in stdout
    assert 'options:' in stdout


def test_invalid_command():
    result = _run_cli(['foo'])
    assert result.returncode == 2
    stderr = result.stderr.strip()
    text = (
        'usage: pyvista [-h] [--version] {report} ...\n'
        "pyvista: error: argument subcommand: invalid choice: 'foo' (choose from report)"
    )
    assert text in stderr


def test_bad_kwarg():
    result = _run_cli(['report', 'foo'])
    assert result.returncode == 1
    stderr = result.stderr.strip()
    assert stderr.endswith("ValueError: Invalid kwarg format: 'foo', expected key=value")


PY_KWARGS = {'gpu': False, 'sort': True}


@pytest.mark.parametrize(
    ('cli_kwargs', 'py_kwargs'),
    [
        (['gpu=False', 'sort=True'], PY_KWARGS),
        (['gpu=false', 'sort=true'], PY_KWARGS),
        (['gpu=no', 'sort=yes'], PY_KWARGS),
    ],
)
def test_report(cli_kwargs, py_kwargs):
    cli_args = ['report', *cli_kwargs]
    result = _run_cli(cli_args)
    actual = result.stdout.strip()
    expected = str(pv.Report(**py_kwargs)).strip()

    # Remove the second line (Date) from both
    actual_clean = pop_second_line(actual)
    expected_clean = pop_second_line(expected)
    assert actual_clean == expected_clean

    # Try again by calling main directly
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(cli_args)
    actual = buf.getvalue().strip()

    # Remove the second line (Date) from both
    actual_clean = pop_second_line(actual)
    expected_clean = pop_second_line(expected)
    assert actual_clean == expected_clean


@pytest.mark.parametrize('as_script', [True, False])
def test_version(as_script):
    result = _run_cli(['--version'], as_script=as_script)
    actual = result.stdout.strip()
    expected = pv.__version__
    assert actual == f'PyVista {expected}'
