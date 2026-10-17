"""Tests for the ``pyvista build`` command and its interactive wizard."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from rich.console import Console

from pyvista.__main__ import main
from pyvista._build import BuildConfig
from pyvista._build import Builder
from pyvista._build import phases
from pyvista._build import tools

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


def _config(tmp_path: Path, **kwargs) -> BuildConfig:
    defaults = dict(backend='vtk', build_dir=tmp_path / 'vtk-build', dry_run=True, os_name='linux')
    defaults.update(kwargs)
    return BuildConfig(**defaults)


@pytest.fixture
def no_tools(mocker: MockerFixture):
    """Pretend every tool is present and pip is available."""
    mocker.patch.object(tools.shutil, 'which', side_effect=lambda name, **_: f'/usr/bin/{name}')
    mocker.patch.object(tools, 'executable_version', return_value='99.0')
    mocker.patch.object(tools, 'python_build_packages', return_value=[])


def test_cli_requires_backend(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    monkeypatch.delenv('PYVISTA_VTK_BACKEND', raising=False)
    with pytest.raises(SystemExit):
        main('build')
    assert 'pyvista build vtk' in capsys.readouterr().err


@pytest.mark.usefixtures('no_tools')
def test_cli_env_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
):
    monkeypatch.setenv('PYVISTA_VTK_BACKEND', 'vtkmodules')
    monkeypatch.chdir(tmp_path)
    main('build --dry-run --with io,web --without rendering -D X=1')
    out = capsys.readouterr().out
    assert 'Building backend: vtk' in out
    assert str(tmp_path / 'vtk-build') in ''.join(out.split())
    assert '-DX=1' in out


def test_cli_source_and_url_exclusive(capsys: pytest.CaptureFixture):
    with pytest.raises(SystemExit):
        main('build vtk --source . --url x')
    assert 'mutually exclusive' in capsys.readouterr().err


@pytest.mark.usefixtures('no_tools')
def test_interactive_adjust_selection(tmp_path: Path, mocker: MockerFixture):
    """Declining the default selection asks about every group and option in turn."""
    mocker.patch.object(phases.sys.stdin, 'isatty', return_value=True)

    def answer(question, **_):
        if question.startswith('Enable'):
            return 'web' in question
        return True

    def prompt(question, default='', **_):
        return 'n' if question.startswith('Build with these') else default

    mocker.patch.object(phases.Confirm, 'ask', side_effect=answer)
    mocker.patch.object(phases.Prompt, 'ask', side_effect=prompt)
    console = Console(record=True, width=200, force_terminal=False)
    cfg = _config(tmp_path, dry_run=False, ref='v9.7.0')
    builder = Builder(cfg, console)
    mocker.patch.object(builder, 'run')
    mocker.patch.object(builder, 'version_suffix', return_value=None)
    builder.phase_source()
    builder.phase_tools()
    builder.phase_groups()
    assert builder.groups == ['web']
    assert '-DVTK_GROUP_ENABLE_Rendering=NO' in builder.configure_command()
    assert '-DVTK_MODULE_ENABLE_VTK_WebCore=YES' in builder.configure_command()


def test_interactive_required_and_conflicting_groups(tmp_path: Path, mocker: MockerFixture):
    """A required group is never asked about and a conflicting one is skipped."""
    src = tmp_path / 'src'
    (src / '.git').mkdir(parents=True)
    (src / 'pyvista-build.toml').write_text(
        'manifest_version = 1\n'
        '[backend]\nname = "demo"\nimport_name = "demo"\ndefault_ref = "v1"\n'
        '[groups.b]\nhelp = "B"\ndefault = false\ncmake = ["B=ON"]\nconflicts = ["c"]\n'
        '[groups.c]\nhelp = "C"\ndefault = false\ncmake = ["C=ON"]\nconflicts = ["b"]\n'
        '[groups.core]\nhelp = "Core"\nrequired = true\ncmake = ["CORE=ON"]\n'
    )
    mocker.patch.object(phases.sys.stdin, 'isatty', return_value=True)
    asked: list[str] = []

    def answer(question, **_):
        asked.append(question)
        return True

    mocker.patch.object(phases.Confirm, 'ask', side_effect=answer)
    mocker.patch.object(
        phases.Prompt,
        'ask',
        side_effect=lambda q, default='', **_: (
            'n' if q.startswith('Build with these') else default
        ),
    )
    mocker.patch.object(tools.shutil, 'which', side_effect=lambda name, **_: f'/usr/bin/{name}')
    mocker.patch.object(tools, 'executable_version', return_value='99.0')
    mocker.patch.object(tools, 'python_build_packages', return_value=[])
    cfg = _config(tmp_path, backend='demo', source=src, dry_run=False)
    builder = Builder(cfg, Console(quiet=True))
    builder.phase_source()
    builder.phase_tools()
    builder.phase_groups()
    assert builder.groups == ['b', 'core']
    assert not any('core' in q or "'c'" in q for q in asked)


def test_cli_interrupt(mocker: MockerFixture, tmp_path: Path, capsys: pytest.CaptureFixture):
    mocker.patch.object(Builder, 'build', side_effect=KeyboardInterrupt)
    with pytest.raises(SystemExit) as info:
        main(f'build vtk --dry-run --build-dir {tmp_path}')
    assert info.value.code == 130
    captured = capsys.readouterr()
    assert 'Stopped.' in captured.out
    assert 'PyVista build stopped' not in captured.err


@pytest.mark.usefixtures('no_tools')
def test_interactive_adjust_configure(tmp_path: Path, mocker: MockerFixture):
    """Declining the configure settings asks for the build type, suffix and entries."""
    mocker.patch.object(phases.sys.stdin, 'isatty', return_value=True)
    mocker.patch.object(
        phases.Confirm, 'ask', side_effect=lambda q, **_: not q.startswith('Run cmake with')
    )
    answers = iter(['y', 'y', 'Debug', '', 'X=1 Y=2'])
    mocker.patch.object(phases.Prompt, 'ask', side_effect=lambda *_, **__: next(answers))
    console = Console(record=True, width=200, force_terminal=False)
    builder = Builder(_config(tmp_path, dry_run=False, version_suffix='dev0+gabc'), console)
    run = mocker.patch.object(builder, 'run')
    builder.phase_source()
    builder.phase_tools()
    builder.phase_groups()
    builder.phase_configure()
    cmd = run.call_args.args[0]
    assert '-DCMAKE_BUILD_TYPE=Debug' in cmd
    assert '-DVTK_VERSION_SUFFIX=' in cmd
    assert cmd[-2:] == ['-DX=1', '-DY=2']
    out = console.export_text()
    assert 'dev0+gabc' in out
    assert '(release)' in out


@pytest.mark.usefixtures('no_tools')
def test_interactive_source_prompt(tmp_path: Path, mocker: MockerFixture):
    """The clone prompt takes a fork URL, a local checkout, or a plain yes or no."""
    mocker.patch.object(phases.sys.stdin, 'isatty', return_value=True)
    console = Console(record=True, width=200, force_terminal=False)
    builder = Builder(_config(tmp_path, dry_run=False), console)
    git = mocker.patch.object(builder, 'git', return_value='')
    answers = iter(['nonsense', 'https://example.com/me/vtk.git', 'mybranch'])
    mocker.patch.object(phases.Prompt, 'ask', side_effect=lambda *_, **__: next(answers))
    builder.phase_source()
    assert 'neither a git checkout nor a git URL' in console.export_text()
    assert git.call_args_list[0].args[2] == 'https://example.com/me/vtk.git'
    assert git.call_args_list[1].args[-1] == 'mybranch'

    fork = tmp_path / 'fork'
    (fork / '.git').mkdir(parents=True)
    builder = Builder(_config(tmp_path, dry_run=False), Console(quiet=True))
    mocker.patch.object(phases.Prompt, 'ask', return_value=str(fork))
    builder.phase_source()
    assert builder.config.source == fork.resolve()
    assert builder.manifest.name == 'vtk'

    builder = Builder(_config(tmp_path, dry_run=False), Console(quiet=True))
    mocker.patch.object(phases.Prompt, 'ask', return_value='n')
    with pytest.raises(phases.BuildError, match='Nothing to build'):
        builder.phase_source()


@pytest.mark.usefixtures('no_tools')
def test_interactive_selection_flags(tmp_path: Path, mocker: MockerFixture):
    """Flags typed at the groups prompt change the selection in place."""
    mocker.patch.object(phases.sys.stdin, 'isatty', return_value=True)
    answers = iter(
        [
            'y',
            '--bogus',
            '--without',
            '--without serialization,web --opt smp=TBB',
            '--with web --opt smp=Sequential',
            '--minimal --with=io',
            '',
        ]
    )
    mocker.patch.object(
        phases.Prompt, 'ask', side_effect=lambda *_, default='', **__: next(answers) or default
    )
    console = Console(record=True, width=200, force_terminal=False)
    builder = Builder(_config(tmp_path, dry_run=False, ref='v9.7.0'), console)
    mocker.patch.object(builder, 'run')
    builder.phase_source()
    builder.phase_tools()
    builder.phase_groups()
    out = console.export_text()
    assert "Unknown flag '--bogus'" in out
    assert '--without needs a value' in out
    assert builder.groups == ['io', 'web']
    assert builder.option_values['smp'] == 'Sequential'
    assert builder.config.without == ['serialization']
