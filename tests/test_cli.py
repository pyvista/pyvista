from __future__ import annotations

import contextlib
import io
import subprocess
import sys

import numpy as np
import pytest

import pyvista as pv
from pyvista.__main__ import COMMANDS_URL
from pyvista.__main__ import main


def pop_second_line(text: str) -> str:
    """Return text with the second line removed."""
    lines = text.splitlines()
    if len(lines) > 1:
        lines.pop(1)
    return '\n'.join(lines)


def _run_pyvista(argv: list[str] | None = None, *, as_script: bool = False):
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
    result = _run_pyvista(args)
    assert result.returncode == 1
    stdout = result.stdout

    assert stdout.startswith('usage: pyvista [-h] [--version] {report,plot} ...')
    assert 'positional arguments:' in stdout
    assert 'options:' in stdout


@pytest.mark.skipif(sys.version_info < (3, 12), reason='Different output format on older python')
def test_invalid_command():
    result = _run_pyvista(['foo'])
    assert result.returncode == 2
    stderr = result.stderr.strip()
    text = (
        'usage: pyvista [-h] [--version] {report,plot} ...\n'
        "pyvista: error: argument subcommand: invalid choice: 'foo'"
    )
    assert text in stderr


def test_bad_kwarg():
    result = _run_pyvista(['report', 'foo=bar'])
    assert result.returncode == 1
    stderr = result.stderr.strip()
    assert stderr.endswith("TypeError: Report.__init__() got an unexpected keyword argument 'foo'")


def test_missing_required_arg():
    result = _run_pyvista(['plot'])
    assert result.returncode == 1
    stderr = result.stderr.strip()
    assert stderr.endswith("TypeError: plot() missing 1 required positional argument: 'var_item'")


def test_arg_after_kwarg():
    result = _run_pyvista(['report', 'foo=bar', 'foo'])
    assert result.returncode == 1
    stderr = result.stderr.strip()
    assert stderr.endswith(
        'SyntaxError: Positional argument foo must not follow a keyword argument.'
    )


PY_KWARGS = {'gpu': False, 'sort': True}


@pytest.mark.parametrize(
    ('cli_kwargs', 'py_kwargs'),
    [
        (['gpu=False', 'sort=True'], PY_KWARGS),
        (['gpu=false', 'sort=true'], PY_KWARGS),
        (['gpu=no', 'sort=yes'], PY_KWARGS),
        (['gpu=no', 'sort=foo'], PY_KWARGS),
    ],
)
def test_report(cli_kwargs, py_kwargs):
    cli_args = ['report', *cli_kwargs]
    result = _run_pyvista(cli_args)
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


def test_report_help():
    result = _run_pyvista(['report', '-h'])
    url = COMMANDS_URL['report']
    assert url.startswith('https')
    epilog = f'See documentation for available arguments and keywords:\n{url}'
    assert epilog in result.stdout


@pytest.mark.parametrize('as_script', [True, False])
def test_version(as_script):
    result = _run_pyvista(['--version'], as_script=as_script)
    actual = result.stdout.strip()
    expected = pv.__version__
    assert actual == f'pyvista {expected}'


@pytest.mark.needs_vtk_version(9, 4, 0, reason='bad X server connection')
@pytest.mark.parametrize('background', [[0, 0, 0], [255, 255, 255]])
def test_plot(background):
    file = pv.examples.antfile
    result = _run_pyvista(
        ['plot', file, 'off_screen=True', 'screenshot=True', f'background={background}']
    )
    assert str(np.array(background)) in result.stdout
