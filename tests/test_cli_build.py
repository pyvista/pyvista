"""Tests for the ``pyvista build`` command and its interactive wizard."""

from __future__ import annotations

from dataclasses import fields
import subprocess
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
    mocker.patch.object(tools, 'command_succeeds', return_value=True)
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
    assert 'To build a fork instead' not in out


def test_cli_source_and_url_exclusive(capsys: pytest.CaptureFixture):
    with pytest.raises(SystemExit):
        main('build vtk --source . --url x')
    assert 'mutually exclusive' in capsys.readouterr().err


@pytest.mark.usefixtures('no_tools')
def test_interactive_adjust_selection(tmp_path: Path, mocker: MockerFixture):
    """Declining the default selection asks about every group and option in turn."""
    mocker.patch.object(phases.sys.stdin, 'isatty', return_value=True)

    def answer(question, **_):
        return 'web' in question

    walked: list[bool] = []

    def prompt(question, default='', **_):
        if question.startswith('Build with these'):
            walked.append(True)
            return 'y' if len(walked) > 1 else 'n'
        return default

    mocker.patch.object(phases.Confirm, 'ask', side_effect=answer)
    mocker.patch.object(phases.Prompt, 'ask', side_effect=prompt)
    console = Console(record=True, width=4000, force_terminal=False)
    cfg = _config(tmp_path, dry_run=False, interactive=True, ref='v9.7.0')
    builder = Builder(cfg, console)
    mocker.patch.object(builder, 'run')
    mocker.patch.object(builder, 'version_suffix', return_value=None)
    builder.phase_source()
    builder.phase_tools()
    builder.phase_groups()
    assert builder.groups == ['web']
    assert '-DVTK_GROUP_ENABLE_Rendering=NO' in builder.configure_command()
    assert '-DVTK_MODULE_ENABLE_VTK_WebCore=WANT' in builder.configure_command()


def test_interactive_required_and_conflicting_groups(tmp_path: Path, mocker: MockerFixture):
    """A required group is never asked about and a conflicting one is skipped."""
    src = tmp_path / 'src'
    (src / '.git').mkdir(parents=True)
    (src / 'pyvista-build.toml').write_text(
        'manifest_version = 1\n'
        '[backend]\nname = "demo"\ndefault_ref = "v1"\n'
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
    walked: list[bool] = []

    def prompt(*_, **__):
        walked.append(True)
        return 'y' if len(walked) > 1 else 'n'

    mocker.patch.object(phases.Prompt, 'ask', side_effect=prompt)
    mocker.patch.object(tools.shutil, 'which', side_effect=lambda name, **_: f'/usr/bin/{name}')
    mocker.patch.object(tools, 'executable_version', return_value='99.0')
    mocker.patch.object(tools, 'python_build_packages', return_value=[])
    cfg = _config(tmp_path, backend='demo', source=src, dry_run=False, interactive=True)
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
def test_interactive_configure_flags(tmp_path: Path, mocker: MockerFixture):
    """Flags typed at the configure prompt change the cmake command in place."""
    mocker.patch.object(phases.sys.stdin, 'isatty', return_value=True)
    answers = iter(
        [
            'y',
            'y',
            '--build-type Fast',
            '--compiler-cache turbo',
            '--build-type Debug --version-suffix "" -D X=1 --define X=2 -D Y=3',
            '--compiler-cache none',
            '--compiler-cache ccache -D Z=1 --build-type Fast',
            '',
        ]
    )
    mocker.patch.object(
        phases.Prompt, 'ask', side_effect=lambda *_, default='', **__: next(answers) or default
    )
    ccache = tools.Tool('ccache', ('ccache',), path='/usr/bin/ccache')
    mocker.patch.object(tools, 'compiler_cache_tool', return_value=ccache)
    console = Console(record=True, width=4000, force_terminal=False)
    builder = Builder(
        _config(tmp_path, dry_run=False, interactive=True, version_suffix='dev0+gabc'), console
    )
    run = mocker.patch.object(builder, 'run')
    builder.phase_source()
    builder.phase_tools()
    builder.phase_groups()
    builder.phase_configure()
    cmd = run.call_args.args[0]
    assert '-DCMAKE_BUILD_TYPE=Debug' in cmd
    assert '-DVTK_VERSION_SUFFIX=' in cmd
    assert cmd[-2:] == ['-DX=2', '-DY=3']
    assert '-DCMAKE_C_COMPILER_LAUNCHER=' in cmd
    assert builder.launcher is None
    assert builder.config.compiler_cache == 'none'
    assert 'CCACHE_BASEDIR' not in builder.env
    out = console.export_text()
    assert out.count('--build-type must be one of') == 2
    assert '--compiler-cache must be one of' in out
    assert 'dev0+gabc' in out
    assert '(release)' in out


@pytest.mark.usefixtures('no_tools')
def test_interactive_configure_walk(tmp_path: Path, mocker: MockerFixture):
    """Answering n at the configure prompt asks for each setting in turn."""
    mocker.patch.object(phases.sys.stdin, 'isatty', return_value=True)
    answers = iter(['y', 'y', 'n', 'RelWithDebInfo', 'ccache', 'dev0+gfeed', 'A=1 B=2', 'y'])
    mocker.patch.object(phases.Prompt, 'ask', side_effect=lambda *_, **__: next(answers))
    builder = Builder(_config(tmp_path, dry_run=False, interactive=True), Console(quiet=True))
    run = mocker.patch.object(builder, 'run')
    builder.phase_source()
    builder.phase_tools()
    builder.phase_groups()
    builder.phase_configure()
    cmd = run.call_args.args[0]
    assert '-DCMAKE_BUILD_TYPE=RelWithDebInfo' in cmd
    assert '-DVTK_VERSION_SUFFIX=dev0+gfeed' in cmd
    assert cmd[-2:] == ['-DA=1', '-DB=2']
    assert builder.launcher is not None
    assert builder.launcher.name == 'ccache'
    assert '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache' in cmd


@pytest.mark.usefixtures('no_tools')
def test_interactive_source_prompt(tmp_path: Path, mocker: MockerFixture):
    """The clone prompt takes a fork URL, a local checkout, or a plain yes or no."""
    mocker.patch.object(phases.sys.stdin, 'isatty', return_value=True)
    console = Console(record=True, width=4000, force_terminal=False)
    builder = Builder(_config(tmp_path, dry_run=False, interactive=True), console)
    git = mocker.patch.object(builder, 'git', return_value='')
    answers = iter(['nonsense', 'https://example.com/me/vtk.git', 'mybranch'])
    mocker.patch.object(phases.Prompt, 'ask', side_effect=lambda *_, **__: next(answers))
    builder.phase_source()
    out = console.export_text()
    assert 'neither a git checkout nor a git URL' in out
    assert 'To build a fork instead' in out
    assert git.call_args_list[0].args[3] == 'https://example.com/me/vtk.git'
    assert git.call_args_list[1].args[-1] == 'mybranch'

    fork = tmp_path / 'fork'
    (fork / '.git').mkdir(parents=True)
    builder = Builder(_config(tmp_path, dry_run=False, interactive=True), Console(quiet=True))
    mocker.patch.object(phases.Prompt, 'ask', return_value=str(fork))
    builder.phase_source()
    assert builder.config.source == fork.resolve()
    assert builder.manifest.name == 'vtk'

    builder = Builder(_config(tmp_path, dry_run=False, interactive=True), Console(quiet=True))
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
    console = Console(record=True, width=4000, force_terminal=False)
    builder = Builder(_config(tmp_path, dry_run=False, interactive=True, ref='v9.7.0'), console)
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


@pytest.mark.usefixtures('no_tools')
def test_interactive_reuse_prompt(tmp_path: Path, mocker: MockerFixture):
    """An existing checkout in the build tree is confirmed, replaced, or swapped for another."""
    mocker.patch.object(phases.sys.stdin, 'isatty', return_value=True)
    cfg = _config(tmp_path, dry_run=False, interactive=True)
    (cfg.src_dir / '.git').mkdir(parents=True)
    console = Console(record=True, width=4000, force_terminal=False)
    builder = Builder(cfg, console)
    git = mocker.patch.object(builder, 'git', return_value='')
    mocker.patch.object(phases.Prompt, 'ask', return_value='y')
    builder.phase_source()
    assert 'Reusing the checkout' in console.export_text()
    assert not git.called

    fork = tmp_path / 'fork'
    (fork / '.git').mkdir(parents=True)
    builder = Builder(_config(tmp_path, dry_run=False, interactive=True), Console(quiet=True))
    mocker.patch.object(phases.Prompt, 'ask', return_value=str(fork))
    builder.phase_source()
    assert builder.config.source == fork.resolve()

    builder = Builder(_config(tmp_path, dry_run=False, interactive=True), Console(quiet=True))
    git = mocker.patch.object(builder, 'git', return_value='')
    mocker.patch.object(
        phases.Prompt, 'ask', side_effect=['https://example.com/me/vtk.git', 'dev']
    )
    mocker.patch.object(phases.Confirm, 'ask', return_value=False)
    with pytest.raises(phases.BuildError, match='Keeping the existing checkout'):
        builder.phase_source()
    assert (cfg.src_dir / '.git').exists()

    cfg_url = _config(tmp_path, dry_run=False, interactive=True, url='https://example.com/f.git')
    builder = Builder(cfg_url, Console(quiet=True))
    mocker.patch.object(builder, 'git', return_value='https://example.com/vtk.git')
    prompt = mocker.patch.object(phases.Prompt, 'ask', return_value='y')
    with pytest.raises(phases.BuildError, match='--url is not used'):
        builder.phase_source()
    assert not prompt.called

    builder = Builder(_config(tmp_path, dry_run=False, interactive=True), Console(quiet=True))
    git = mocker.patch.object(builder, 'git', return_value='')
    mocker.patch.object(
        phases.Prompt, 'ask', side_effect=['https://example.com/me/vtk.git', 'dev']
    )
    mocker.patch.object(phases.Confirm, 'ask', return_value=True)
    builder.phase_source()
    assert not cfg.src_dir.exists()
    assert git.call_args_list[0].args[3] == 'https://example.com/me/vtk.git'
    assert git.call_args_list[1].args[-1] == 'dev'


@pytest.mark.usefixtures('no_tools')
def test_source_dir_that_is_not_a_checkout(tmp_path: Path, mocker: MockerFixture):
    """A non-empty ``<build-dir>/src`` without git is kept; an empty one is cloned into."""
    cfg = _config(tmp_path, dry_run=False)
    cfg.src_dir.mkdir(parents=True)
    (cfg.src_dir / 'pyproject.toml').write_text('')
    builder = Builder(cfg, Console(quiet=True))
    git = mocker.patch.object(builder, 'git', return_value='')
    with pytest.raises(phases.BuildError, match='exists but is not a git checkout'):
        builder.phase_source()
    assert (cfg.src_dir / 'pyproject.toml').exists()
    assert not git.called
    (cfg.src_dir / 'pyproject.toml').unlink()
    builder.phase_source()
    assert git.call_args_list[0].args[0] == 'clone'


@pytest.mark.usefixtures('no_tools')
def test_url_with_existing_checkout_stops(tmp_path: Path, mocker: MockerFixture):
    """A ``--url`` the existing checkout was not cloned from stops before anything runs."""
    cfg = _config(tmp_path, url='https://example.com/f.git')
    (cfg.src_dir / '.git').mkdir(parents=True)
    builder = Builder(cfg, Console(quiet=True))
    git = mocker.patch.object(builder, 'git', return_value='https://example.com/vtk.git')
    with pytest.raises(
        phases.BuildError, match=r'checkout of https://example\.com/vtk\.git already'
    ):
        builder.plan()
    with pytest.raises(phases.BuildError, match='--url is not used'):
        builder.phase_source()
    for interactive in (False, True):
        mocker.patch.object(phases.sys.stdin, 'isatty', return_value=interactive)
        cfg.dry_run, cfg.interactive = not interactive, interactive
        builder.console = Console(record=True, width=4000, force_terminal=False)
        with pytest.raises(phases.BuildError, match='--url is not used'):
            builder.build()
        assert 'steps' not in builder.console.export_text()
    git.side_effect = subprocess.CalledProcessError(2, 'git')
    with pytest.raises(phases.BuildError, match='checkout of another URL'):
        builder.plan()
    git.side_effect = None
    for origin in (cfg.url, 'https://example.com/f', 'https://example.com/f/'):
        git.return_value = origin
        assert builder._unused_url() is None
    cfg.url = 'https://example.com/f/'
    git.return_value = 'https://example.com/f.git'
    assert builder._unused_url() is None


def test_parse_flags_inline_values():
    valued = {'-D', '--define', '--version-suffix'}
    pairs = Builder._parse_flags('-DA=1 --version-suffix= --define=B=2 -D C=3', set(), valued)
    assert pairs == [('-D', 'A=1'), ('--version-suffix', ''), ('--define', 'B=2'), ('-D', 'C=3')]


def test_parse_flags_windows_paths(monkeypatch: pytest.MonkeyPatch):
    """On Windows, backslashes in typed paths survive and surrounding quotes are dropped."""
    text = r'-D Qt5_DIR=C:\Qt\5.15\lib -D "X=C:\Program Files\x" -D Y="C:\a b"'
    monkeypatch.setattr(phases.os, 'name', 'nt')
    pairs = Builder._parse_flags(text, set(), {'-D'})
    monkeypatch.undo()
    assert pairs == [
        ('-D', r'Qt5_DIR=C:\Qt\5.15\lib'),
        ('-D', r'X=C:\Program Files\x'),
        ('-D', r'Y=C:\a b'),
    ]


def test_configure_table_lists_only_prompt_flags(tmp_path: Path, mocker: MockerFixture):
    """Every flag in the configure table is one the configure prompt accepts."""
    console = Console(record=True, width=4000, force_terminal=False)
    build_dir = tmp_path / 'm[red]x[/red]y'
    builder = Builder(_config(tmp_path, build_dir=build_dir, version_suffix='[b]q'), console)
    builder._manifest = phases.find_manifest('vtk', None)
    printed = mocker.spy(builder.console, 'print')
    builder._print_configure_settings()
    out = console.export_text()
    assert str(build_dir / 'build') in out
    assert '[b]q' in out
    flags = [str(cell) for cell in printed.call_args.args[0].columns[2]._cells if str(cell)]
    assert '--build-dir' not in flags
    errors = []
    for flag in flags:
        try:
            builder._apply_configure_flags(f'{flag} x' if ' ' not in flag else flag)
        except phases.BuildError as err:
            errors.append(str(err))
    assert not [e for e in errors if 'Unknown flag' in e]


def test_configure_walk_keeps_spaces_in_defines(tmp_path: Path, mocker: MockerFixture):
    cfg = _config(tmp_path, compiler_cache='none', version_suffix='')
    builder = Builder(cfg, Console(quiet=True))
    builder._manifest = phases.find_manifest('vtk', None)
    answers = iter(
        [
            'Release',
            'none',
            '',
            'X="a b" Y=2',
            'Release',
            'none',
            '-',
            '"X=1',
            'Debug',
            'none',
            '-',
            '-',
        ]
    )
    defaults = []

    def ask(*_, default='', **__):
        defaults.append(default)
        return next(answers)

    mocker.patch.object(phases.Prompt, 'ask', side_effect=ask)
    builder._walk_configure_settings()
    assert cfg.defines == ['X=a b', 'Y=2']
    with pytest.raises(phases.BuildError, match='Cannot parse'):
        builder._walk_configure_settings()
    assert defaults[-1] == "'X=a b' Y=2"
    assert cfg.defines == ['X=a b', 'Y=2']
    builder._walk_configure_settings()
    assert cfg.defines == []
    assert cfg.version_suffix == ''
    assert defaults[-2] == '-'


def test_prompts_print_brackets_verbatim():
    """Prompt text keeps its [y/n] hints and is not markup."""
    console = Console(record=True, width=4000, force_terminal=False)
    prompt = phases.Prompt('Build with these? [y/n, or --with]', console=console)
    assert '[y/n, or --with]' in str(prompt.make_prompt('y'))
    assert str(prompt.make_prompt('y')).endswith('(y): ')
    confirm = phases.Confirm('Delete it? [careful]', console=console)
    assert '[careful]' in str(confirm.make_prompt(False))


@pytest.fixture
def captured(mocker: MockerFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Record the configuration each ``pyvista build`` call would build with."""
    monkeypatch.chdir(tmp_path)
    configs: list[BuildConfig] = []

    def fake_build(self):
        configs.append(self.config)

    mocker.patch.object(Builder, 'build', fake_build)
    return configs


def test_cli_defaults(captured: list[BuildConfig], tmp_path: Path):
    main('build vtk')
    expected = BuildConfig(backend='vtk', build_dir=(tmp_path / 'vtk-build').resolve())
    assert captured[-1] == expected


@pytest.mark.parametrize(
    ('args', 'field', 'value'),
    [
        ('--no-repair', 'repair', False),
        ('--compiler-cache auto', 'compiler_cache', 'auto'),
        ('--build-type Debug', 'build_type', 'Debug'),
        ('--jobs 3', 'jobs', 3),
        ('--clean', 'clean', True),
        ('--reconfigure', 'reconfigure', True),
        ('--update', 'update', True),
        ('--ref v9.6.0', 'ref', 'v9.6.0'),
        ('--opt smp=TBB --opt web=OFF', 'options', ['smp=TBB', 'web=OFF']),
        ('--version-suffix dev1', 'version_suffix', 'dev1'),
        ('-D A=1 --define B=2', 'defines', ['A=1', 'B=2']),
        ('--with io,web --with qt', 'with_', ['io', 'web', 'qt']),
        ('--without web', 'without', ['web']),
        ('--minimal', 'minimal', True),
        ('--url https://example.com/f.git', 'url', 'https://example.com/f.git'),
        ('--yes', 'yes', True),
        ('-y', 'yes', True),
        ('--interactive', 'interactive', True),
        ('-i', 'interactive', True),
        ('--dry-run', 'dry_run', True),
        ('--verbose', 'verbose', True),
        ('-v', 'verbose', True),
        ('--install', 'install', True),
    ],
)
def test_cli_flag_sets_config(captured: list[BuildConfig], args: str, field: str, value):
    """Each flag sets its own configuration field and leaves every other at its default."""
    main('build vtk')
    default = captured[-1]
    main(f'build vtk {args}')
    assert getattr(captured[-1], field) == value
    changed = [
        f.name
        for f in fields(BuildConfig)
        if getattr(captured[-1], f.name) != getattr(default, f.name)
    ]
    assert changed == [field]


def test_cli_path_flags(captured: list[BuildConfig], tmp_path: Path):
    """Path flags are resolved against the working directory."""
    (tmp_path / 'checkout').mkdir()
    main('build vtk --build-dir out --compiler-cache ccache --cache-dir cache --source checkout')
    assert captured[-1].build_dir == (tmp_path / 'out').resolve()
    assert captured[-1].cache_dir == (tmp_path / 'cache').resolve()
    assert captured[-1].source == (tmp_path / 'checkout').resolve()


def test_cli_manylinux(captured: list[BuildConfig]):
    main('build vtk --dry-run')
    assert captured[-1].manylinux is None
    main('build vtk --dry-run --manylinux')
    assert captured[-1].manylinux.startswith('quay.io/pypa/manylinux_2_28_')
    main('build vtk --dry-run --manylinux --manylinux-image my/image')
    assert captured[-1].manylinux == 'my/image'
    main('build vtk --dry-run -i')
    assert captured[-1].interactive is True
    assert captured[-1].yes is False
    main('build vtk --dry-run --yes')
    assert captured[-1].yes is True
    main('build vtk --dry-run -y --install')
    assert (captured[-1].yes, captured[-1].install) == (True, True)


def test_cli_manylinux_image_needs_manylinux(capsys: pytest.CaptureFixture):
    with pytest.raises(SystemExit):
        main('build vtk --dry-run --manylinux-image my/image')
    assert '--manylinux-image needs --manylinux' in capsys.readouterr().err


def test_cli_cache_dir_needs_compiler_cache(capsys: pytest.CaptureFixture):
    with pytest.raises(SystemExit):
        main('build vtk --dry-run --cache-dir cache')
    assert '--cache-dir needs --compiler-cache' in capsys.readouterr().err


def test_cli_os_error(mocker: MockerFixture, tmp_path: Path, capsys: pytest.CaptureFixture):
    mocker.patch.object(Builder, 'build', side_effect=PermissionError('denied: [x]'))
    with pytest.raises(SystemExit):
        main(f'build vtk --dry-run --build-dir {tmp_path}')
    err = capsys.readouterr().err
    assert 'PyVista build stopped' in err
    assert 'denied: [x]' in err


def test_cli_end_of_input(mocker: MockerFixture, tmp_path: Path):
    mocker.patch.object(Builder, 'build', side_effect=EOFError)
    with pytest.raises(SystemExit) as info:
        main(f'build vtk --dry-run --build-dir {tmp_path}')
    assert info.value.code == 130


def test_cli_build_error(mocker: MockerFixture, tmp_path: Path, capsys: pytest.CaptureFixture):
    mocker.patch.object(Builder, 'build', side_effect=phases.BuildError('no [bold]compiler[/x]'))
    with pytest.raises(SystemExit):
        main(f'build vtk --dry-run --build-dir {tmp_path}')
    err = capsys.readouterr().err
    assert 'PyVista build stopped' in err
    assert 'no [bold]compiler[/x]' in err


def test_default_ref_without_shipped_manifest(tmp_path: Path):
    builder = Builder(_config(tmp_path, backend='demo'), Console(quiet=True))
    assert builder._default_ref() is None


def test_fork_ref_defaults_to_its_manifest(tmp_path: Path, mocker: MockerFixture):
    """Enter at the ref prompt leaves the ref to the cloned fork's own manifest."""
    mocker.patch.object(phases.sys.stdin, 'isatty', return_value=True)
    cfg = _config(tmp_path, backend='cvista', interactive=True, dry_run=False)
    builder = Builder(cfg, Console(quiet=True))
    ask = mocker.patch.object(
        phases.Prompt, 'ask', side_effect=['https://example.com/me/cvista.git', '']
    )
    assert builder._ask_source('Clone it?', stop='stopped') == (None, True)
    assert builder.config.ref is None
    assert "default ref in the fork's manifest" in ask.call_args.args[0]
    assert ask.call_args.kwargs['show_default'] is False


def test_prompt_entries_are_checked(tmp_path: Path, mocker: MockerFixture):
    """Prompt answers take - for a release version and -D entries with or without -D."""
    builder = Builder(_config(tmp_path, version_suffix='dev0+gabc'), Console(quiet=True))
    builder._apply_configure_flags('--version-suffix - -D -DQ=1')
    assert builder.config.version_suffix == ''
    assert builder.config.defines == ['Q=1']
    with pytest.raises(phases.BuildError, match="KEY=VALUE or KEY:TYPE=VALUE, got 'FOO'"):
        builder._apply_configure_flags('-D FOO')
    answers = iter(['', '', '-', '-DA=1 B:BOOL=ON'])
    mocker.patch.object(
        phases.Prompt, 'ask', side_effect=lambda *_, default='', **__: next(answers) or default
    )
    builder._walk_configure_settings()
    assert builder.config.defines == ['A=1', 'B:BOOL=ON']
    answers = iter(['', '', '-', 'NOVALUE'])
    with pytest.raises(phases.BuildError, match='KEY=VALUE'):
        builder._walk_configure_settings()
    builder = Builder(_config(tmp_path, defines=['-DX=1', 'FOO']), Console(quiet=True))
    with pytest.raises(phases.BuildError, match="got 'FOO'"):
        builder.plan()


@pytest.mark.usefixtures('no_tools')
def test_build_tree_from_another_source(tmp_path: Path, mocker: MockerFixture):
    """A build tree configured from another checkout is replaced only on a yes."""
    cfg = _config(tmp_path, dry_run=False, yes=True)
    cfg.cmake_dir.mkdir(parents=True)
    other = tmp_path / 'other-vtk'
    cache = cfg.cmake_dir / 'CMakeCache.txt'
    cache.write_text(f'# comment\nCMAKE_HOME_DIRECTORY:INTERNAL={other}\n')
    console = Console(record=True, width=4000, force_terminal=False)
    builder = Builder(cfg, console)
    run = mocker.patch.object(builder, 'run')
    builder.phase_source()
    builder.phase_tools()
    builder.phase_groups()
    with pytest.raises(phases.BuildError, match='Pass --clean to replace it'):
        builder.phase_configure()
    assert f'was configured from {other}' in console.export_text()
    assert cache.exists()

    mocker.patch.object(builder, 'confirm', return_value=True)
    builder.phase_configure()
    assert not cache.exists()
    assert run.call_args.args[0][:2] == ['cmake', '-Wno-dev']

    cfg.cmake_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(f'CMAKE_HOME_DIRECTORY:INTERNAL={cfg.src_dir}\n')
    assert builder._configured_source() == cfg.src_dir
    cache.write_text('CMAKE_BUILD_TYPE:STRING=Release\n')
    assert builder._configured_source() is None

    cache.write_text(f'CMAKE_HOME_DIRECTORY:INTERNAL={other}\n')
    builder.config.dry_run = True
    builder.phase_configure()
    assert cache.exists()


def test_existing_checkout_declined(tmp_path: Path, mocker: MockerFixture):
    src = tmp_path / 'src'
    src.mkdir()
    builder = Builder(_config(tmp_path, dry_run=False, ref='v1'), Console(quiet=True))
    git = mocker.patch.object(
        builder, 'git', side_effect=lambda *a, **_: a[-1] if a[0] == 'rev-parse' else ''
    )
    mocker.patch.object(builder, 'confirm', return_value=False)
    builder._existing_checkout(src)
    assert not any('checkout' in call.args for call in git.call_args_list)


@pytest.mark.usefixtures('no_tools')
def test_interactive_selection_parse_errors(tmp_path: Path, mocker: MockerFixture):
    """An unbalanced quote or a clashing flag is reported and leaves the selection as it was."""
    mocker.patch.object(phases.sys.stdin, 'isatty', return_value=True)
    answers = iter(['y', '--with "web', '--with nosuch', ''])
    mocker.patch.object(
        phases.Prompt, 'ask', side_effect=lambda *_, default='', **__: next(answers) or default
    )
    console = Console(record=True, width=4000, force_terminal=False)
    builder = Builder(_config(tmp_path, dry_run=False, interactive=True, ref='v9.7.0'), console)
    mocker.patch.object(builder, 'run')
    builder.phase_source()
    builder.phase_tools()
    default_groups = phases.resolve_groups(builder.manifest, 'linux')
    builder.phase_groups()
    out = console.export_text()
    assert 'Cannot parse' in out
    assert 'Unknown group' in out
    assert not builder.config.minimal
    assert builder.groups == default_groups


@pytest.mark.usefixtures('no_tools')
def test_interactive_selection_minimal(tmp_path: Path, mocker: MockerFixture):
    """Typing --minimal at the selection prompt keeps only the required groups."""
    mocker.patch.object(phases.sys.stdin, 'isatty', return_value=True)
    answers = iter(['y', '--minimal', ''])
    mocker.patch.object(
        phases.Prompt, 'ask', side_effect=lambda *_, default='', **__: next(answers) or default
    )
    builder = Builder(
        _config(tmp_path, dry_run=False, interactive=True, ref='v9.7.0'), Console(quiet=True)
    )
    mocker.patch.object(builder, 'run')
    builder.phase_source()
    builder.phase_groups()
    assert builder.config.minimal
    assert builder.groups == phases.resolve_groups(builder.manifest, 'linux', minimal=True)


def _walk_answers(smp: str, answers: list[str]):
    """Return a ``Prompt.ask`` stand-in: ``smp`` for the smp option, else the next answer."""
    queue = iter(answers)
    return lambda question='', **_: smp if question.startswith('smp') else next(queue)


@pytest.mark.usefixtures('no_tools')
def test_walk_writes_the_configuration(tmp_path: Path, mocker: MockerFixture):
    """A bad walk answer keeps the old selection; a good one survives later flags."""
    console = Console(record=True, width=4000, force_terminal=False)
    builder = Builder(_config(tmp_path, ref='v9.7.0'), console)
    builder.phase_source()
    builder.phase_groups()
    before = list(builder.groups)
    mocker.patch.object(phases.Confirm, 'ask', return_value=True)
    mocker.patch.object(phases.Prompt, 'ask', side_effect=_walk_answers('bogus', ['n', 'y']))
    ask = dict(
        apply=builder._apply_selection_flags,
        resolve=builder._resolve_selection,
        show=builder._print_selection,
        walk=builder._adjust_selection,
    )
    builder._ask_until_yes('Build?', **ask)
    assert 'bogus' in console.export_text()
    assert (builder.config.options, builder.config.minimal) == ([], False)
    assert builder.groups == before
    mocker.patch.object(phases.Prompt, 'ask', side_effect=_walk_answers('TBB', ['n', 'y']))
    builder._ask_until_yes('Build?', **ask)
    assert builder.config.options == ['smp=TBB']
    assert 'headless' in builder.groups
    builder._apply_selection_flags('--without web')
    builder._resolve_selection()
    assert builder.option_values['smp'] == 'TBB'
    assert 'web' not in builder.groups
    assert 'headless' in builder.groups


@pytest.mark.usefixtures('no_tools')
def test_interactive_clean_declined_and_cache_kept(tmp_path: Path, mocker: MockerFixture):
    """Declining --clean keeps the build tree, and keeping the cache choice changes nothing."""
    mocker.patch.object(phases.sys.stdin, 'isatty', return_value=True)
    cfg = _config(tmp_path, dry_run=False, interactive=True, clean=True, compiler_cache='none')
    cfg.cmake_dir.mkdir(parents=True)
    (cfg.cmake_dir / 'kept').write_text('')
    answers = iter(['y', 'y', 'n', 'Release', 'none', '', '', 'y'])
    mocker.patch.object(phases.Prompt, 'ask', side_effect=lambda *_, **__: next(answers))
    mocker.patch.object(phases.Confirm, 'ask', return_value=False)
    builder = Builder(cfg, Console(quiet=True))
    mocker.patch.object(builder, 'run')
    builder.phase_source()
    builder.phase_tools()
    builder.phase_groups()
    builder.phase_configure()
    assert (cfg.cmake_dir / 'kept').exists()
    assert cfg.compiler_cache == 'none'
