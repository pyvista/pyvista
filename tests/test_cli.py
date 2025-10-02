from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import textwrap

import pytest
from rich.console import Console

import pyvista as pv
from pyvista.__main__ import app
from pyvista.__main__ import main


@pytest.fixture
def console():
    return Console(
        width=70,
        force_terminal=True,
        highlight=False,
        color_system=None,
        legacy_windows=False,
    )


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
    assert stdout.startswith('usage: pyvista [-h] [--version] {report} ...')
    assert 'positional arguments:' in stdout
    assert 'options:' in stdout


@pytest.mark.skipif(sys.version_info < (3, 12), reason='Different output format on older python')
def test_invalid_command():
    result = _run_pyvista(['foo'])
    assert result.returncode == 2
    stderr = result.stderr.strip()
    text = (
        'usage: pyvista [-h] [--version] {report} ...\n'
        "pyvista: error: argument subcommand: invalid choice: 'foo'"
    )
    assert text in stderr


def test_bad_kwarg():
    result = _run_pyvista(['report', 'foo'])
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


def test_report_help(capsys: pytest.CaptureFixture, console: Console):
    app('report --help', console=console)

    assert (
        textwrap.dedent(
            """
            ╭─ Parameters ───────────────────────────────────────────────────────╮
            │ ADDITIONAL --additional  List of packages or package names to add  │
            │                          to output information.                    │
            │ NCOL --ncol              Number of package-columns in html table;  │
            │                          only has effect if                        │
            │                          ``mode='HTML'`` or ``mode='html'``.       │
            │                          [default: 3]                              │
            │ TEXT-WIDTH --text-width  The text width for non-HTML display       │
            │                          modes. [default: 80]                      │
            │ SORT --sort --no-sort    Alphabetically sort the packages.         │
            │                          [default: False]                          │
            │ GPU --gpu --no-gpu       Gather information about the GPU.         │
            │                          Defaults to ``True`` but if               │
            │                          experiencing rendering issues, pass       │
            │                          ``False`` to safely generate a            │
            │                          report. [default: True]                   │
            │ DOWNLOADS --downloads    Gather information about downloads. If    │
            │   --no-downloads         ``True``, includes:                       │
            │                          - The local user data path (where         │
            │                          downloads are saved)                      │
            │                          - The VTK Data source (where files are    │
            │                          downloaded from)                          │
            │                          - Whether local file caching is enabled   │
            │                          for the VTK Data source                   │
            │                                                                    │
            │                          .. versionadded:: 0.47 [default: False]   │
            ╰────────────────────────────────────────────────────────────────────╯
        """
        )
        in capsys.readouterr().out
    )


def test_version(capsys: pytest.CaptureFixture):
    main('--version')
    assert capsys.readouterr().out == f'pyvista {pv.__version__}\n'


def test_help(capsys: pytest.CaptureFixture, console: Console):
    app('--help', console=console)

    assert (
        textwrap.dedent(
            """
        ╭─ Commands ─────────────────────────────────────────────────────────╮
        │ report     Generate a PyVista software environment report.         │
        │ --help -h  Display this message and exit.                          │
        │ --version  Display application version.                            │
        ╰────────────────────────────────────────────────────────────────────╯
            """
        )
        in capsys.readouterr().out
    )
