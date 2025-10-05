from __future__ import annotations

import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING

import pytest
from pytest_cases import parametrize
from pytest_cases import parametrize_with_cases
from rich.console import Console

import pyvista as pv
from pyvista.__main__ import app
from pyvista.__main__ import main

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from pytest_mock import MockerFixture


@pytest.fixture
def patch_app_console(monkeypatch: pytest.MonkeyPatch):
    console = Console(
        width=70,
        force_terminal=True,
        highlight=False,
        color_system=None,
        legacy_windows=False,
    )
    monkeypatch.setattr(app, 'console', console)


@pytest.mark.parametrize('args', [[], ''])
@pytest.mark.usefixtures('patch_app_console')
def test_no_input(args, capsys: pytest.CaptureFixture):
    main(args)

    expected = textwrap.dedent(
        """\
        Usage: pyvista COMMAND

        ╭─ Commands ─────────────────────────────────────────────────────────╮
        │ report     Generate a PyVista software environment report.         │
        │ --help -h  Display this message and exit.                          │
        │ --version  Display application version.                            │
        ╰────────────────────────────────────────────────────────────────────╯
        """
    )
    assert expected == capsys.readouterr().out


@pytest.mark.usefixtures('patch_app_console')
def test_invalid_command(capsys: pytest.CaptureFixture):
    expected = textwrap.dedent(
        """\
    Usage: pyvista COMMAND

    ╭─ Commands ─────────────────────────────────────────────────────────╮
    │ report     Generate a PyVista software environment report.         │
    │ --help -h  Display this message and exit.                          │
    │ --version  Display application version.                            │
    ╰────────────────────────────────────────────────────────────────────╯
    ╭─ Error ────────────────────────────────────────────────────────────╮
    │ Unknown command "foo". Available commands: report.                 │
    ╰────────────────────────────────────────────────────────────────────╯
    """
    )
    with pytest.raises(SystemExit) as e:
        main('foo')
    assert e.value.code == 1
    assert expected == capsys.readouterr().out


@pytest.mark.usefixtures('patch_app_console')
def test_bad_kwarg_report(capsys: pytest.CaptureFixture):
    expected = textwrap.dedent(
        """\
    Usage: pyvista report [ARGS] [OPTIONS]

    Generate a PyVista software environment report.

    .. versionadded:: 0.47

        The report can now be generated using the shell command:

        .. code-block:: shell

            pyvista report --sort ...

        Run ``pyvista report --help`` for more details on available
    parameters.

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
    ╭─ Error ────────────────────────────────────────────────────────────╮
    │ Unknown option: "--foo=1".                                         │
    ╰────────────────────────────────────────────────────────────────────╯
    """
    )
    with pytest.raises(SystemExit) as e:
        main('report --foo=1')
    assert e.value.code == 1
    assert expected == capsys.readouterr().out


@pytest.fixture
def mock_report(mocker: MockerFixture):
    return mocker.patch.object(pv.__main__, 'Report')


class CasesReport:
    def case_empty(self):
        return '', (), {}

    @parametrize(downloads=['True', 'yes', 'y', 'true'])
    @parametrize(sort=['True', 'yes', 'y', 'true'])
    def case_kw_bool(self, downloads, sort):
        return f'--downloads={downloads} --sort={sort}', (), dict(downloads=True, sort=True)

    @parametrize(downloads=['False', 'no', 'n', 'false'])
    @parametrize(sort=['False', 'no', 'n', 'false'])
    def case_kw_bool_no(self, downloads, sort):
        return f'--downloads={downloads} --sort={sort}', (), dict(downloads=False, sort=False)

    def case_bool(self):
        return '--downloads --sort', (), dict(downloads=True, sort=True)

    def case_no_bool(self):
        return '--no-downloads --no-sort', (), dict(downloads=False, sort=False)

    def case_additional(self):
        return '--additional "foo"', (['foo'],), {}

    def case_additional_multiple_kw(self):
        return '--additional "foo" --additional "bar"', (['foo', 'bar'],), {}

    def case_additional_multiple_args(self):
        return '"foo" "bar"', (['foo', 'bar'],), {}

    def case_additional_ncol(self):
        return '"foo" --ncol 2', (['foo'], 2), {}

    def case_additional_textwidth(self):
        # `textwidth` is keyword whereas `additional` is positional since inspect.BoundArguments
        # enforces it
        return '"foo" --text-width 100', (['foo'],), dict(text_width=100)


@parametrize_with_cases('tokens, expected_args, expected_kwargs', cases=CasesReport)
def test_report_called(
    tokens: str,
    expected_args: tuple,
    expected_kwargs: dict,
    mock_report: MagicMock,
):
    """Test that the Report class is called with the expected arguments."""
    main(f'report {tokens}')
    mock_report.assert_called_once_with(*expected_args, **expected_kwargs)


@pytest.mark.usefixtures('patch_app_console')
def test_report_help(capsys: pytest.CaptureFixture):
    main('report --help')

    expected = textwrap.dedent(
        """\
            Usage: pyvista report [ARGS] [OPTIONS]

            Generate a PyVista software environment report.
       """
    )
    assert expected in capsys.readouterr().out


def test_version(capsys: pytest.CaptureFixture):
    main('--version')
    assert capsys.readouterr().out == f'pyvista {pv.__version__}\n'


@pytest.mark.usefixtures('patch_app_console')
def test_help(capsys: pytest.CaptureFixture):
    main('--help')

    expected = textwrap.dedent(
        """\
        Usage: pyvista COMMAND

        ╭─ Commands ─────────────────────────────────────────────────────────╮
        │ report     Generate a PyVista software environment report.         │
        │ --help -h  Display this message and exit.                          │
        │ --version  Display application version.                            │
        ╰────────────────────────────────────────────────────────────────────╯
        """
    )
    assert expected == capsys.readouterr().out


@parametrize(
    as_script=[True, False],
    tokens_err_codes=[
        ('--foo', 1),
        ('report --foo', 1),
        ('', 0),
        ('report --help', 0),
        ('--help', 0),
    ],
)
def test_cli_entry_point(as_script: bool, tokens_err_codes: tuple[str, int]):
    args, exit_code_expected = tokens_err_codes

    args = f'{sys.executable} -m pyvista ' + args if not as_script else 'pyvista ' + args
    process = subprocess.run(
        args,
        check=False,
        capture_output=True,
        encoding='utf-8',
    )

    assert process.returncode == exit_code_expected
