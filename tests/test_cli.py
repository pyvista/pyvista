from __future__ import annotations

import inspect
import shlex
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING

import pytest
from pytest_cases import case
from pytest_cases import filters
from pytest_cases import fixture
from pytest_cases import parametrize
from pytest_cases import parametrize_with_cases
from rich.console import Console

import pyvista as pv
from pyvista.__main__ import app
from pyvista.__main__ import main

if TYPE_CHECKING:
    from pathlib import Path
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
        │ plot       Plot a PyVista, numpy, or vtk object.                   │
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
    │ plot       Plot a PyVista, numpy, or vtk object.                   │
    │ report     Generate a PyVista software environment report.         │
    │ --help -h  Display this message and exit.                          │
    │ --version  Display application version.                            │
    ╰────────────────────────────────────────────────────────────────────╯
    ╭─ Error ────────────────────────────────────────────────────────────╮
    │ Unknown command "foo". Available commands: report, plot.           │
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
    │   --empty-additional     to output information.                    │
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
    """  # noqa: W291, RUF100
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


@pytest.fixture
def mock_plot(mocker: MockerFixture):
    return mocker.patch.object(pv, 'plot')


@pytest.fixture
def mock_files_validator(mocker: MockerFixture):
    mocker.patch.object(pv.__main__, '_validator_files')


@fixture
def default_plot_kwargs():
    return {
        'var_item': [],
        'anti_aliasing': None,
        'background': None,
        'border': False,
        'border_color': 'k',
        'border_width': 2.0,
        'eye_dome_lighting': False,
        'full_screen': None,
        'interactive': True,
        'notebook': None,
        'off_screen': None,
        'parallel_projection': False,
        'return_cpos': False,
        'screenshot': None,
        'show_axes': None,
        'show_bounds': False,
        'ssao': False,
        'text': '',
        'volume': False,
        'window_size': None,
        'zoom': None,
    }


class CasesPlot:
    def case_empty(self, default_plot_kwargs: dict):
        return '', default_plot_kwargs

    @pytest.mark.usefixtures('mock_files_validator')
    def case_files_single_args(self, default_plot_kwargs: dict):
        """Test when only a single positional argument is given for files."""
        kwargs = default_plot_kwargs
        kwargs['var_item'] = [f := 'file.vtp']
        return f, kwargs

    @pytest.mark.usefixtures('mock_files_validator')
    def case_files_multiple_args(self, default_plot_kwargs: dict):
        """Test when multiple positional arguments are given for files."""
        kwargs = default_plot_kwargs
        kwargs['var_item'] = (files := ['file1.vtp', 'file2.vtp'])
        return ' '.join(files), kwargs

    @parametrize(with_space=[True, False])
    @pytest.mark.usefixtures('mock_files_validator')
    def case_files_multiple_kargs(self, default_plot_kwargs: dict, with_space: bool):
        """Test when multiple keyword arguments are given for files."""
        kwargs = default_plot_kwargs
        kwargs['var_item'] = (files := ['file1.vtp', 'file2.vtp'])
        prefix = '--files ' if with_space else '--files='
        return prefix + ' '.join(files), kwargs

    @parametrize(offscreen=['True', 'yes', 'y', 'true'])
    def case_kw_bool(self, default_plot_kwargs: dict, offscreen: str):
        kwargs = default_plot_kwargs
        tokens = f'--off-screen={offscreen}'
        kwargs.update(off_screen=True)
        return tokens, kwargs

    @parametrize(offscreen=['False', 'no', 'n', 'false'])
    def case_kw_no_bool(self, default_plot_kwargs: dict, offscreen: str):
        kwargs = default_plot_kwargs
        tokens = f'--off-screen={offscreen}'
        kwargs.update(off_screen=False)
        return tokens, kwargs

    @parametrize(off_screen=[True, False])
    def case_kw_no_bool_no_value(self, default_plot_kwargs: dict, off_screen: bool):
        kwargs = default_plot_kwargs
        tokens = '--off-screen' if off_screen else '--no-off-screen'
        kwargs.update(off_screen=off_screen)
        return tokens, kwargs

    def case_window_size(self, default_plot_kwargs: dict):
        kwargs = default_plot_kwargs
        tokens = '--window-size=[100,100]'
        kwargs.update(window_size=[100, 100])
        return tokens, kwargs

    def case_window_size_multiple(self, default_plot_kwargs: dict):
        kwargs = default_plot_kwargs
        tokens = '--window-size 100 100'
        kwargs.update(window_size=[100, 100])
        return tokens, kwargs

    def case_window_size_rounding(self, default_plot_kwargs: dict):
        """Test when window size is given as float, it is rounded to int."""
        kwargs = default_plot_kwargs
        tokens = '--window-size=[100.4,100.6]'
        kwargs.update(window_size=[100, 101])
        return tokens, kwargs

    @parametrize(anti_aliasing=['ssaa', 'msaa', 'fxaa'])
    def case_anti_aliasing(self, default_plot_kwargs: dict, anti_aliasing: str):
        kwargs = default_plot_kwargs
        tokens = f'--anti-aliasing={anti_aliasing}'
        kwargs.update(anti_aliasing=anti_aliasing)
        return tokens, kwargs

    @parametrize(
        kwargs=[
            ('--color=red', dict(color='red')),
            ('--color=(0,1,0)', dict(color=(0, 1, 0))),
            ('--color=[0,1,0]', dict(color=[0, 1, 0])),
            ('--color=[0.1,1,0]', dict(color=[0.1, 1, 0])),
            ('--clim [0.1,1]', dict(clim=[0.1, 1])),
        ]
    )
    def case_kwargs(self, default_plot_kwargs: dict, kwargs: tuple[str, dict]):
        """Test when kwargs are provided to Plotter.add_mesh"""
        tokens, kwargs = kwargs
        default_plot_kwargs.update(**kwargs)
        return tokens, default_plot_kwargs

    @case(tags='raises')
    def case_anti_aliasing_raises(self):
        return '--anti-aliasing=foo'

    @case(tags='raises')
    @parametrize(window_size=['100', '100 200 300', '[100,200,300]'])
    def case_window_size_wrong_length(self, window_size: str):
        """Test when the window size does not have exactly two elements."""
        return f' --window-size {window_size}'

    @case(tags='raises')
    @parametrize(window_size=['100 a', 'b a', '[a,b]'])
    def case_window_size_wrong_type(self, window_size: str):
        """Test when the window size does not have the correct type."""
        return f'--window-size {window_size}'

    @case(tags='raises')
    def case_files_raises(self, tmp_path: Path):
        """Test when the file does not exists."""
        return str(tmp_path / 'file.vtp')

    @case(tags='raises')
    def case_files_raises_kw(self, tmp_path: Path):
        """Test when the file does not exists as keyword."""
        return f'--files={tmp_path / "file.vtp"}'

    @case(tags='raises')
    def case_files_raises_one_exists(self, tmp_path: Path):
        """Test when one file does not exists."""
        (f1 := (tmp_path / 'f1.vtp')).touch()
        return f'--files {f1.as_posix()} {(tmp_path / "f2.vtp").as_posix()}'

    @case(tags='raises')
    def case_files_raises_not_readable(self, tmp_path: Path, mocker: MockerFixture):
        """Test when a file is not readable by pyvista"""
        (f1 := (tmp_path / 'f1.vtp')).touch()
        m = mocker.patch.object(pv, 'read')
        m.side_effect = Exception('Not readable')

        return f'--files {f1.as_posix()}'

    @case(tags='raises')
    def case_kw_contains_hyphen(self):
        """Test when a supplementary keyword argument is given with an hyphen"""
        return '--foo-bar bar'


@parametrize_with_cases(
    'tokens, expected_kwargs',
    cases=CasesPlot,
    filter=~filters.has_tag('raises'),
)
def test_plot_called(
    tokens: str,
    expected_kwargs: dict,
    mock_plot: MagicMock,
):
    """Test that the pv.plot function is called with the expected arguments."""
    main(f'plot {tokens}')
    mock_plot.assert_called_once_with(**expected_kwargs)


@parametrize_with_cases('tokens', cases=CasesPlot, has_tag='raises')
@pytest.mark.usefixtures('mock_plot')
def test_plot_called_raises(tokens: str):
    """Test that the plot CLI is raising expected exit errors."""
    with pytest.raises(SystemExit) as e:
        main(f'plot {tokens}')

    assert e.value.code == 1


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


@pytest.mark.usefixtures('patch_app_console')
def test_plot_help(capsys: pytest.CaptureFixture):
    main('plot --help')

    expected = textwrap.dedent(
        """\
            Usage: pyvista plot [ARGS] [OPTIONS]

            Plot a PyVista, numpy, or vtk object.
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
        │ plot       Plot a PyVista, numpy, or vtk object.                   │
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
        ('plot --foo', 1),
        ('', 0),
        ('report --help', 0),
        ('plot --help', 0),
        ('--help', 0),
    ],
)
def test_cli_entry_point(as_script: bool, tokens_err_codes: tuple[str, int]):
    args = [sys.executable, '-m', 'pyvista'] if not as_script else ['pyvista']

    argv, exit_code_expected = tokens_err_codes
    args += [*shlex.split(argv)]

    process = subprocess.run(
        args,
        check=False,
        capture_output=True,
        encoding='utf-8',
    )

    assert process.returncode == exit_code_expected


def test_plot_signature_subset():
    """
    Since the `pyvista plot` CLI exposes a subset of the original `pv.plot` arguments,
    any changes made in the signature of `pv.plot` must be accounted (or not) in the
    `pyvista plot` CLI.

    This test will fail if the argument names are different.
    """
    sig = set(inspect.signature(pv.plot).parameters.keys())
    sig_sub = set(inspect.signature(pv.__main__._plot).parameters.keys())

    allowed_missing = {
        'jupyter_backend',
        'theme',
        'var_item',
        'return_viewer',
        'return_img',
        'cpos',
        'jupyter_kwargs',
    }
    diff = sig - sig_sub - allowed_missing
    assert diff == set()
