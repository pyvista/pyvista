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
from pytest_cases import get_case_tags
from pytest_cases import parametrize
from pytest_cases import parametrize_with_cases
from rich.console import Console

import pyvista as pv
from pyvista.__main__ import app
from pyvista.__main__ import main

if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock

    from pytest_cases.case_parametrizer_new import Case
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
    ╭─ Error ────────────────────────────────────────────────────────────╮
    │ Unknown option: "--foo=1".                                         │
    ╰────────────────────────────────────────────────────────────────────╯
    """
    )
    with pytest.raises(SystemExit) as e:
        main('report --foo=1')
    assert e.value.code == 1
    assert expected == '\n'.join(capsys.readouterr().out.split('\n')[-4:])


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
def mock_plotter(mocker: MockerFixture):
    return mocker.patch.object(pv, 'Plotter')


@pytest.fixture
def mock_plot(mocker: MockerFixture):
    return mocker.patch.object(pv, 'plot')


@pytest.fixture
def mock_add_mesh(mock_plotter: MagicMock):
    return mock_plotter().add_mesh


@pytest.fixture
def mock_add_volume(mock_plotter: MagicMock):
    return mock_plotter().add_volume


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
            ('--color="(0, 1, 0)"', dict(color=(0, 1, 0))),
            ('--color="[0, 1, 0]"', dict(color=[0, 1, 0])),
            ('--color=[0,1,0]', dict(color=[0, 1, 0])),
            ('--color=[0.1,1,0]', dict(color=[0.1, 1, 0])),
            ('--clim [0.1,1]', dict(clim=[0.1, 1])),
            ('--clim [0.1,1] --color red', dict(clim=[0.1, 1], color='red')),
        ]
    )
    @pytest.mark.usefixtures('mock_files_validator')
    @case(tags=['kwargs', 'add_mesh'])
    def case_kwargs(self, default_plot_kwargs: dict, kwargs: tuple[str, dict]):
        """Test when kwargs are provided to Plotter.add_mesh"""
        tokens, kwargs = kwargs
        tokens += ' --files=file.vtp'
        default_plot_kwargs.update(**kwargs)
        default_plot_kwargs.update(var_item=['file.vtp'])
        return tokens, default_plot_kwargs

    @parametrize(
        kwargs=[
            ('--mapper smart', dict(mapper='smart')),
            ('--mapper smart --blending additive', dict(mapper='smart', blending='additive')),
        ]
    )
    @pytest.mark.usefixtures('mock_files_validator')
    @case(tags=['kwargs', 'add_volume'])
    def case_kwargs_volume(self, default_plot_kwargs: dict, kwargs: tuple[str, dict]):
        """Test when kwargs are provided to Plotter.add_volume"""
        tokens, kwargs = kwargs
        tokens += ' --files=file.vtp --volume'
        default_plot_kwargs.update(**kwargs, volume=True)
        default_plot_kwargs.update(var_item=['file.vtp'])
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
    def case_kw_unknown(self, tmp_path: Path):
        """Test when a supplementary keyword argument does not exists"""
        pv.Sphere().save(f := tmp_path / 'file.vtp')
        return f'{f.as_posix()} --foo_bar bar'

    @case(tags='raises')
    def case_wrong_argument(self, tmp_path: Path):
        """Test when an argument raises an error"""
        pv.Sphere().save(f := tmp_path / 'file.vtp')
        return f'{f.as_posix()} --opacity=foo'


@parametrize_with_cases(
    'tokens, expected_kwargs',
    cases=CasesPlot,
    filter=~filters.has_tag('raises'),
)
def test_plot_called(
    tokens: str,
    expected_kwargs: dict,
    mock_plot: MagicMock,
    mock_add_mesh: MagicMock,
    mock_add_volume: MagicMock,
    mocker: MockerFixture,
    current_cases: dict[str, Case],
):
    """Test that the pv.plot function is called with the expected arguments."""
    main(f'plot {tokens}')
    mock_plot.assert_called_once_with(**expected_kwargs)

    case = current_cases['tokens']
    if 'kwargs' in (tags := get_case_tags(case_func=case.func)):
        mocker.stop(mock_plot)
        main(f'plot {tokens}')

        mock = mock_add_mesh if 'add_mesh' in tags else mock_add_volume
        kwargs = case.params['kwargs'][-1]
        mock.assert_called_once_with('file.vtp', **kwargs)


@parametrize(
    tokens_ncalls_args=[
        ('file1.vtp file2.vtp', 2, ['file1.vtp', 'file2.vtp']),
        ('', 0, []),
        ('--files file1.vtp file2.vtp file3.vtp', 3, ['file1.vtp', 'file2.vtp', 'file3.vtp']),
    ],
    idgen=lambda **args: args['tokens_ncalls_args'][0],
)
@parametrize(func=['add_mesh', 'add_volume'])
@pytest.mark.usefixtures('mock_files_validator')
def test_add_mesh_volume_called(
    tokens_ncalls_args: tuple[str, int, list[str]],
    mock_add_mesh: MagicMock,
    mock_add_volume: MagicMock,
    mocker: MockerFixture,
    func: str,
):
    """Test that the pv.Plotter.add_mesh and add_volume methods are called with the expected arguments."""  # noqa: E501
    tokens, ncalls, args = tokens_ncalls_args
    tokens += ' --volume' if (add_volume := (func == 'add_volume')) else ''
    main(f'plot {tokens}')

    mock = mock_add_volume if add_volume else mock_add_mesh
    assert mock.call_count == ncalls
    assert mock.mock_calls == [mocker.call(a) for a in args]


@parametrize_with_cases('tokens', cases=CasesPlot, has_tag='raises')
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
    assert expected == '\n'.join(capsys.readouterr().out.split('\n')[:4])


@pytest.mark.usefixtures('patch_app_console')
def test_plot_help(capsys: pytest.CaptureFixture):
    main('plot --help')

    expected = textwrap.dedent(
        """\
        Usage: pyvista plot file (file2) [OPTIONS]

        Plot a PyVista, numpy, or vtk object.
        """
    )
    assert expected == '\n'.join(capsys.readouterr().out.split('\n')[:4])


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
        'notebook',
    }
    diff = sig - sig_sub - allowed_missing
    assert diff == set()


@parametrize(func=['plot', 'report'])
@parametrize(ret=['foo', None])
def test_print(
    mock_plot: MagicMock,
    mock_report: MagicMock,
    func: str,
    capsys: pytest.CaptureFixture,
    ret: str | None,
):
    """Test that the output of the functions are sent to stdout."""
    mock = mock_plot if func == 'plot' else mock_report
    mock.return_value = ret
    main(f'{func}')

    expected = f'{ret}\n' if ret is not None else ''
    assert capsys.readouterr().out == expected
