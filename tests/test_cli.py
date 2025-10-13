from __future__ import annotations

import inspect
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING
from typing import Any

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
        │ convert    Convert a mesh file to another format.                  │
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
    │ convert    Convert a mesh file to another format.                  │
    │ plot       Plot a PyVista, numpy, or vtk object.                   │
    │ report     Generate a PyVista software environment report.         │
    │ --help -h  Display this message and exit.                          │
    │ --version  Display application version.                            │
    ╰────────────────────────────────────────────────────────────────────╯
    ╭─ Error ────────────────────────────────────────────────────────────╮
    │ Unknown command "foo". Available commands: report, convert, plot.  │
    ╰────────────────────────────────────────────────────────────────────╯
    """
    )
    with pytest.raises(SystemExit) as e:
        main('foo')
    assert e.value.code == 1
    assert expected == capsys.readouterr().out


@pytest.mark.usefixtures('patch_app_console')
@pytest.mark.parametrize('command', ['report', 'convert'])
def test_bad_kwarg_command(capsys: pytest.CaptureFixture, command):
    expected = textwrap.dedent(
        """\
    ╭─ Error ────────────────────────────────────────────────────────────╮
    │ Unknown option: "--foo=1".                                         │
    ╰────────────────────────────────────────────────────────────────────╯
    """
    )
    with pytest.raises(SystemExit) as e:
        main(f'{command} --foo=1')
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


class CasesConvert:
    """CLI argument parsing cases for `convert`."""

    def case_new_ext(self):
        return 'ant.ply new.vtp', 'new.vtp'

    def case_same_ext(self):
        return 'ant.ply ant.ply', 'ant.ply'

    def case_dir(self):
        return 'ant.ply foo/new.vtp', 'foo/new.vtp'

    def case_nested_dir(self):
        return 'ant.ply foo/bar/new.vtp', 'foo/bar/new.vtp'

    def case_wildcard_ext(self):
        return 'ant.ply .vtp', 'ant.vtp'

    def case_wildcard_dir(self):
        return 'ant.ply bar/.vtp', 'bar/ant.vtp'


@pytest.fixture
def tmp_ant_file(tmp_path):
    """Return a temp path to the ant file for tests."""
    src = Path(pv.examples.antfile)
    dst = tmp_path / src.name
    shutil.copy(src, dst)

    # Change cwd to tmp_path temporarily
    old_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        yield dst
    finally:
        os.chdir(old_cwd)


@parametrize_with_cases('tokens, expected_file', cases=CasesConvert)
def test_convert_called(tokens, expected_file, tmp_ant_file):  # noqa: ARG001
    main(shlex.split(f'convert {tokens}', posix=True))
    assert Path(expected_file).is_file()


@pytest.mark.usefixtures('patch_app_console')
def test_convert_dir_only_error(tmp_ant_file: Path, capsys: pytest.CaptureFixture):
    with pytest.raises(SystemExit) as e:
        main(f'convert {str(tmp_ant_file)!r} {str(tmp_ant_file.parent)!r}')

    out = capsys.readouterr().out
    assert '╭─ Error ─' in out, out
    assert 'Invalid value' in out, out
    assert 'Output file must have a file extension.' in out, out
    assert e.value.code == 1


@pytest.mark.usefixtures('patch_app_console')
def test_convert_file_not_found(capsys: pytest.CaptureFixture):
    file_in = 'missing.vtp'
    with pytest.raises(SystemExit) as e:
        main(f'convert {file_in} .ply')
    out = capsys.readouterr().out
    assert '╭─ Error ─' in out, out
    assert f'1 file not found: {file_in}' in out, out
    assert e.value.code == 1


@pytest.mark.usefixtures('patch_app_console')
def test_convert_read_error(tmp_path: Path, capsys: pytest.CaptureFixture):
    # Create a dummy .vtp file with empty contents
    name = 'dummy.vtp'
    file_in = tmp_path / name
    file_in.write_text('')
    assert file_in.is_file()

    with pytest.raises(SystemExit) as e:
        main(f'convert {str(file_in)!r} .ply')

    out = capsys.readouterr().out
    assert '╭─ Error ─' in out, out
    assert '1 file not readable by PyVista:' in out, out
    assert e.value.code == 1


@pytest.mark.usefixtures('patch_app_console')
def test_convert_save_error(tmp_ant_file: Path, capsys: pytest.CaptureFixture):
    invalid_suffix = '.foo'
    output_path = tmp_ant_file.with_suffix(invalid_suffix)
    with pytest.raises(SystemExit) as e:
        main(f'convert {str(tmp_ant_file)!r} {str(output_path)!r}')

    out = capsys.readouterr().out
    assert '╭─ PyVista Error ─' in out, out
    assert 'Failed to save output file: ' in out, out
    assert output_path.name in out, out
    assert 'Invalid file extension' in out, out
    assert e.value.code == 1


@pytest.mark.usefixtures('patch_app_console')
def test_convert_help(capsys: pytest.CaptureFixture):
    main('convert --help')

    expected = textwrap.dedent(
        """\
            Usage: pyvista convert FILE-IN FILE-OUT

            Convert a mesh file to another format.

            Sample usage:
              $ pyvista convert foo.abc bar.xyz
              Saved: bar.xyz

              $ pyvista convert foo.abc .xyz
              Saved: foo.xyz
       """
    )
    assert expected == '\n'.join(capsys.readouterr().out.split('\n')[:11])


@pytest.fixture
def mock_plotter(mocker: MockerFixture):
    return mocker.patch.object(pv, 'Plotter')


@pytest.fixture
def mock_plot(mocker: MockerFixture):
    return mocker.patch.object(pv, 'plot')


@pytest.fixture
def mock_pv_read(mocker: MockerFixture):
    return mocker.patch.object(pv, 'read')


@pytest.fixture
def mock_add_mesh(mock_plotter: MagicMock):
    return mock_plotter().add_mesh


@pytest.fixture
def mock_add_volume(mock_plotter: MagicMock):
    return mock_plotter().add_volume


@fixture
def missing_plot_arguments():
    """Argument names in the `pv.plot` signature which are intentionally removed from the
    `pv.__main__._plot` function
    """

    return {
        'jupyter_backend',
        'theme',
        'return_viewer',
        'return_img',
        'cpos',
        'jupyter_kwargs',
        'notebook',
    }


@fixture
def default_plot_kwargs(missing_plot_arguments: set[str]) -> dict[str, Any]:
    """Default arguments of `pv.plot`."""

    params = inspect.signature(pv.plot).parameters

    return {
        p: v.default
        for p, v in params.items()
        if (p not in missing_plot_arguments)
        and (v.kind != v.VAR_KEYWORD)
        and v.default is not v.empty
    }


def test_plot_cli_synced(missing_plot_arguments: set[str]):
    """
    Since the `pyvista plot` CLI exposes a subset of the original `pv.plot` arguments,
    any changes made in the signature of `pv.plot` must be synced (or not) in the
    `pyvista plot` CLI.

    This test will fail if any:
    - argument names
    - default values
    - type annotations

    are different between those functions.
    """
    plot_sig = inspect.signature(pv.plot)
    plot_params = set(plot_sig.parameters.keys())

    # Test the parameters names
    cli_sig = inspect.signature(pv.__main__._plot)
    cli_params = set(cli_sig.parameters.keys())

    diff = plot_params - cli_params - missing_plot_arguments
    assert diff == set(), (
        f'Found unexpected differences {diff} in the CLI plot signature arguments'
    )

    # Test the parameters defaults
    cli_defaults = {name: p.default for name, p in cli_sig.parameters.items()}
    plot_defaults = {name: plot_sig.parameters[name].default for name in cli_sig.parameters}

    assert cli_defaults == plot_defaults

    # Test the parameters annotations

    # Need to import some types such that inspect eval them using locals()
    from typing import Literal  # noqa: F401

    from pyvista.jupyter import JupyterBackendOptions  # noqa: F401
    from pyvista.plotting._typing import CameraPositionOptions  # noqa: F401
    from pyvista.plotting._typing import ColorLike  # noqa: F401
    from pyvista.plotting._typing import PlottableType  # noqa: F401
    from pyvista.plotting.themes import Theme  # noqa: F401

    plot_annotations = inspect.get_annotations(pv.plot, eval_str=True, locals=locals())
    cli_annotations = inspect.get_annotations(pv.__main__._plot, eval_str=True)

    cli_annotations = {
        k: v.__origin__ for k, v in cli_annotations.items() if k not in ['return', 'kwargs']
    }  # get __origin__ since Annotated type
    plot_annotations = {k: v for k, v in plot_annotations.items() if k != 'return'}

    # Filter only the ones from cli
    plot_annotations = {name: plot_annotations[name] for name in cli_annotations}

    # Filter the ones which have intentionally different annotations
    excludes = {'anti_aliasing', 'background', 'border_color', 'var_item', 'screenshot'}

    plot_annotations = {k: v for k, v in plot_annotations.items() if k not in excludes}
    cli_annotations = {k: v for k, v in cli_annotations.items() if k not in excludes}

    assert plot_annotations == cli_annotations


class CasesPlot:
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
    def case_kw_no_bool_no_value(
        self,
        default_plot_kwargs: dict,
        off_screen: bool,
    ):
        kwargs = default_plot_kwargs
        tokens = '--off-screen' if off_screen else '--no-off-screen'
        kwargs.update(off_screen=off_screen)
        return tokens, kwargs

    def case_window_size(
        self,
        default_plot_kwargs: dict,
    ):
        kwargs = default_plot_kwargs
        tokens = '--window-size=[100,100]'
        kwargs.update(window_size=[100, 100])
        return tokens, kwargs

    def case_window_size_multiple(
        self,
        default_plot_kwargs: dict,
    ):
        kwargs = default_plot_kwargs
        tokens = '--window-size 100 100'
        kwargs.update(window_size=[100, 100])
        return tokens, kwargs

    def case_window_size_rounding(
        self,
        default_plot_kwargs: dict,
    ):
        """Test when window size is given as float, it is rounded to int."""
        kwargs = default_plot_kwargs
        tokens = '--window-size=[100.4,100.6]'
        kwargs.update(window_size=[100, 101])
        return tokens, kwargs

    @parametrize(anti_aliasing=['ssaa', 'msaa', 'fxaa'])
    def case_anti_aliasing(
        self,
        default_plot_kwargs: dict,
        anti_aliasing: str,
    ):
        kwargs = default_plot_kwargs
        tokens = f'--anti-aliasing={anti_aliasing}'
        kwargs.update(anti_aliasing=anti_aliasing)
        return tokens, kwargs

    # region kwargs cases
    @parametrize(
        tokens_kwargs=[
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
    @case(tags=['kwargs', 'add_mesh'])
    def case_kwargs_add_mesh(
        self,
        tokens_kwargs: tuple[str, dict],
    ):
        """Test when kwargs are provided to Plotter.add_mesh"""
        tokens, kwargs = tokens_kwargs
        return tokens, kwargs

    @parametrize(
        tokens_kwargs=[
            ('--mapper smart', dict(mapper='smart')),
            ('--mapper smart --blending additive', dict(mapper='smart', blending='additive')),
        ]
    )
    @case(tags=['kwargs', 'add_volume'])
    def case_kwargs_add_volume(
        self,
        tokens_kwargs: tuple[str, dict],
    ):
        """Test when kwargs are provided to Plotter.add_volume"""
        tokens, kwargs = tokens_kwargs
        tokens += ' --volume'
        return tokens, kwargs

    # endregion kwargs cases

    # region raises cases

    @case(tags='raises')
    def case_anti_aliasing_raises(self):
        return '--anti-aliasing=foo', [
            'Invalid value for "--anti-aliasing": unable to convert "foo" into',
            "one of {'ssaa', 'msaa', 'fxaa'}.",
        ]

    @case(tags='raises')
    @parametrize(window_size=['100', '100 200 300', '[100,200,300]'])
    def case_window_size_wrong_length(self, window_size: str):
        """Test when the window size does not have exactly two elements."""
        if window_size == '100':
            errors = [
                'Invalid value "[100]" for "--window-size". Window size must be a',
                'list of two integers.',
            ]
        elif window_size in {'100 200 300', '[100,200,300]'}:
            errors = [
                'Invalid value "[100, 200, 300]" for "--window-size". Window size ',
                'must be a list of two integers.',
            ]

        return f' --window-size {window_size}', errors

    @case(tags='raises')
    @parametrize(window_size=['100 a', 'b a'])
    def case_window_size_wrong_type(self, window_size: str):
        """Test when the window size does not have the correct type."""
        obj = 'a' if window_size == '100 a' else 'b'
        return f'--window-size {window_size}', [
            f'Invalid value for "--window-size": unable to convert "{obj}" into int.',
        ]

    @case(tags='raises')
    def case_kw_unknown(self):
        """Test when a supplementary keyword argument does not exists"""
        return '--foo_bar bar', [
            '⚠ The following exception has been raised when calling pv.plot: ',
            '"foo_bar" is an invalid keyword argument for  ',
            '`_common_arg_parser`  ',
            'Please check the provided arguments.',
        ]

    @case(tags='raises')
    def case_wrong_argument(self):
        """Test when an argument raises an error"""
        return ' --opacity=foo', [
            '⚠ The following exception has been raised when calling pv.plot: ',
            'Opacity transfer function (foo) unknown. Valid options:',
            "'sigmoid_10', 'sigmoid_15', 'sigmoid_20', 'foreground',",
        ]

    # endregion raises cases


@parametrize_with_cases(
    'tokens, expected_kwargs',
    cases=CasesPlot,
    filter=~(filters.has_tag('raises') | filters.has_tag('kwargs')),
)
@pytest.mark.usefixtures('mock_pv_read')
def test_plot_called(
    tokens: str,
    expected_kwargs: dict,
    mock_plot: MagicMock,
):
    """Test that the pv.plot function is called with the expected arguments."""
    file = Path(pv.examples.antfile).as_posix()
    main(f'plot --files={file} {tokens}')
    mock_plot.assert_called_once_with(var_item=[pv.read(file)], **expected_kwargs)


@parametrize_with_cases(
    'tokens, expected_kwargs',
    cases=CasesPlot,
    has_tag='kwargs',
)
def test_plot_called_kwargs(
    tokens,
    expected_kwargs,
    mock_add_mesh: MagicMock,
    mock_add_volume: MagicMock,
    current_cases: dict[str, Case],
):
    """
    Test that the pl.add_mesh or pl.add_volume function is called with the expected arguments
    when supplementary kw arguments are added to the command line.
    """

    file = Path(pv.examples.antfile).as_posix()
    main(f'plot --files {file} {tokens}')

    case = current_cases['tokens']
    tags = get_case_tags(case_func=case.func)

    mock = mock_add_mesh if 'add_mesh' in tags else mock_add_volume
    mock.assert_called_once_with(pv.read(file), **expected_kwargs)


@parametrize_with_cases('tokens, errors', cases=CasesPlot, has_tag='raises')
@pytest.mark.usefixtures('patch_app_console')
def test_plot_called_raises(tokens: str, errors: list[str], capsys: pytest.CaptureFixture):
    """Test that the plot CLI is raising expected exit errors."""

    file = Path(pv.examples.antfile).as_posix()
    with pytest.raises(SystemExit) as e:
        main(f'plot --files {file} {tokens}')

    assert e.value.code == 1

    out = capsys.readouterr().out
    for error in errors:
        assert error in out, out


class CasesPlotFiles:
    """Cases used to test the --files argument of the plot CLI"""

    @pytest.mark.usefixtures('mock_pv_read')
    def case_single_args(self, default_plot_kwargs: dict):
        """Test when only a single positional argument is given for files."""
        kwargs = default_plot_kwargs
        f = Path(pv.examples.antfile).as_posix()
        kwargs['var_item'] = [pv.read(f)]
        return f, kwargs

    @pytest.mark.usefixtures('mock_pv_read')
    def case_multiple_args(self, default_plot_kwargs: dict):
        """Test when multiple positional arguments are given for files."""
        kwargs = default_plot_kwargs
        files = [Path(pv.examples.antfile).as_posix()] * 2
        kwargs['var_item'] = [pv.read(f) for f in files]
        return ' '.join(files), kwargs

    @parametrize(with_space=[True, False])
    @pytest.mark.usefixtures('mock_pv_read')
    def case_multiple_kargs(self, default_plot_kwargs: dict, with_space: bool):
        """Test when multiple keyword arguments are given for files."""
        kwargs = default_plot_kwargs

        files = [Path(pv.examples.antfile).as_posix()] * 2
        kwargs['var_item'] = [pv.read(f) for f in files]

        prefix = '--files ' if with_space else '--files='
        return prefix + ' '.join(files), kwargs

    @case(tags='raises')
    def case_not_exists(self, tmp_path: Path):
        """Test when the file does not exists."""
        return str((tmp_path / 'file.vtp').as_posix()), ['1 file not found']

    @case(tags='raises')
    def case_not_exists2(self, tmp_path: Path):
        """Test when the filed do not exists."""
        files = [str((tmp_path / f'file_{i}.vtp').as_posix()) for i in range(4)]
        return ' '.join(files), ['4 files not found']

    @case(tags='raises')
    def case_not_exists_kw(self, tmp_path: Path):
        """Test when the file does not exists as keyword."""
        return f'--files={(tmp_path / "file.vtp").as_posix()}', ['1 file not found']

    @case(tags='raises')
    def case_empty(self):
        """Test when no files are passed"""
        return '', ['Command "plot" parameter "--files" requires an argument.']

    @case(tags='raises')
    def case_one_exists(self, tmp_path: Path):
        """Test when one file does not exists."""
        (f1 := (tmp_path / 'f1.vtp')).touch()
        return f'--files {f1.as_posix()} {(tmp_path / "f2.vtp").as_posix()}', ['1 file not found']

    @case(tags='raises')
    def case_not_readable(self, tmp_path: Path, mocker: MockerFixture):
        """Test when a file is not readable by pyvista"""
        (f1 := (tmp_path / 'f1.vtp')).touch()
        m = mocker.patch.object(pv, 'read')
        m.side_effect = Exception('Not readable')

        return f'--files {f1.as_posix()}', ['1 file not readable by PyVista:']


@parametrize_with_cases(
    'tokens, expected_kwargs', cases=CasesPlotFiles, filter=~filters.has_tag('raises')
)
def test_plot_called_files(
    tokens: str,
    expected_kwargs: dict,
    mock_plot: MagicMock,
):
    """Test that the pv.plot function is called with the expected arguments
    for the --files argument"""
    main(f'plot {tokens}')
    mock_plot.assert_called_once_with(**expected_kwargs)


@parametrize_with_cases('tokens, errors', cases=CasesPlotFiles, filter=filters.has_tag('raises'))
@pytest.mark.usefixtures('patch_app_console')
def test_plot_files_raises(tokens: str, errors: list[str], capsys: pytest.CaptureFixture):
    """Test that errors are correctly raised for the --files argument"""
    with pytest.raises(SystemExit) as e:
        main(f'plot {tokens}')

    assert e.value.code == 1

    out = capsys.readouterr().out
    for error in errors:
        assert error in out, out


@parametrize(
    tokens_ncalls_args=[
        ('file1.ply file2.ply', 2, ['file1.ply', 'file2.ply']),
        ('--files file1.ply file2.ply file3.ply', 3, ['file1.ply', 'file2.ply', 'file3.ply']),
    ],
    idgen=lambda **args: args['tokens_ncalls_args'][0],
)
@parametrize(func=['add_mesh', 'add_volume'])
@pytest.mark.usefixtures('mock_pv_read')
def test_add_mesh_volume_called(
    tokens_ncalls_args: tuple[str, int, list[str]],
    mock_add_mesh: MagicMock,
    mock_add_volume: MagicMock,
    mocker: MockerFixture,
    func: str,
    tmp_path: Path,
):
    """Test that the pv.Plotter.add_mesh and add_volume methods are called
    a number of expected times with the correct arguments.
    """
    tokens, ncalls, args = tokens_ncalls_args
    tokens += ' --volume' if (add_volume := (func == 'add_volume')) else ''

    for a in args:
        shutil.copy(pv.examples.antfile, tmp_path / a)

    os.chdir(tmp_path)
    main(f'plot {tokens}')

    mock = mock_add_volume if add_volume else mock_add_mesh
    assert mock.call_count == ncalls
    assert mock.mock_calls == [mocker.call(pv.read(a)) for a in args]


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
        │ convert    Convert a mesh file to another format.                  │
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

    tokens = func if func == 'report' else f'{func} --files={Path(pv.examples.antfile).as_posix()}'
    main(tokens)

    expected = f'{ret}\n' if ret is not None else ''
    assert capsys.readouterr().out == expected
