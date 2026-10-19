"""Tests for the backend build engine."""

from __future__ import annotations

from dataclasses import replace
from io import StringIO
import os
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest
from rich.console import Console

from pyvista._build import BuildConfig
from pyvista._build import Builder
from pyvista._build import BuildError
from pyvista._build import phases
from pyvista._build import tools
from pyvista._build.manifest import parse_manifest
from pyvista._build.manifest import resolve_groups
from pyvista._build.manifest import resolve_options
from pyvista._build.registry import DEFAULT_REFS
from pyvista._build.registry import find_manifest
from tests.test_build_manifest import MINIMAL_MANIFEST

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


@pytest.fixture(autouse=True)
def _unknown_memory(mocker: MockerFixture):
    """Make the default job count one per core, whatever memory the test machine has."""
    mocker.patch.object(tools, 'physical_memory', return_value=None)


@pytest.fixture
def demo():
    return parse_manifest(MINIMAL_MANIFEST)


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
    mocker.patch.object(tools, 'command_succeeds', return_value=True)


def _fake_git(head: str, status: str = ' M file.cxx'):
    """Return a ``subprocess.run`` stand-in for a checkout at ``head`` with ``status`` output."""

    def run(cmd, **_):
        out = {'status': status, 'rev-parse': head if cmd[-1] == 'HEAD' else 'b' * 40}
        return subprocess.CompletedProcess(cmd, 0, stdout=out.get(cmd[1], ''), stderr='')

    return run


def _builder(tmp_path: Path, mocker: MockerFixture, **kwargs) -> Builder:
    """Return a builder on the shipped VTK manifest with the source phase done."""
    cfg = _config(tmp_path, dry_run=False, **kwargs)
    builder = Builder(cfg, Console(record=True, width=4000, force_terminal=False))
    builder._manifest = find_manifest('vtk', None)
    builder.groups = resolve_groups(builder.manifest, cfg.os_name)
    builder.option_values = resolve_options(builder.manifest, None)
    mocker.patch.object(builder, 'version_suffix', return_value=None)
    return builder


def _path_style(path: str) -> str:
    """Return the style a narrated POSIX path gets, unlinked where it has no file URI."""
    return 'magenta' if os.name == 'nt' else f'magenta link file://{path}'


@pytest.mark.usefixtures('no_tools')
def test_cloned_tree_without_manifest(tmp_path: Path, mocker: MockerFixture):
    cfg = _config(tmp_path, backend='demo', url='https://example.invalid/demo.git', dry_run=False)
    builder = Builder(cfg, Console(quiet=True))
    mocker.patch.object(
        builder, 'git', side_effect=lambda *_a, **_k: cfg.src_dir.mkdir(parents=True)
    )
    with pytest.raises(BuildError, match=r'No pyvista-build\.toml'):
        builder.phase_source()


@pytest.mark.usefixtures('no_tools')
def test_clone_checks_out_the_cloned_default_ref(tmp_path: Path, mocker: MockerFixture):
    cfg = _config(tmp_path, backend='demo', url='https://example.invalid/demo.git', dry_run=False)
    builder = Builder(cfg, Console(quiet=True))

    def git(*args, **_):
        if args[0] == 'clone':
            cfg.src_dir.mkdir(parents=True)
            (cfg.src_dir / 'pyvista-build.toml').write_text(
                'manifest_version = 1\n[backend]\nname = "demo"\ndefault_ref = "v1"\n'
            )

    calls = mocker.patch.object(builder, 'git', side_effect=git)
    builder.phase_source()
    assert calls.call_args_list[-1].args == ('-c', 'advice.detachedHead=false', 'checkout', 'v1')


def test_version_command_failure(tmp_path: Path, mocker: MockerFixture):
    src = tmp_path / 'src'
    (src / '.git').mkdir(parents=True)
    (src / 'pyvista-build.toml').write_text(
        'manifest_version = 1\n'
        '[backend]\nname = "demo"\ndefault_ref = "v1"\n'
        'version_command = "{python} version.py"\n'
    )
    mocker.patch.object(tools.shutil, 'which', side_effect=lambda name, **_: f'/usr/bin/{name}')
    cfg = _config(tmp_path, backend='demo', source=src, dry_run=False)
    builder = Builder(cfg, Console(quiet=True))
    builder.phase_source()
    mocker.patch.object(
        subprocess, 'run', side_effect=subprocess.CalledProcessError(3, ['x'], stderr='boom')
    )
    with pytest.raises(BuildError, match='version_command failed with exit status 3: boom'):
        builder.version_suffix()


def test_hook_expansion_keeps_windows_paths(tmp_path: Path, mocker: MockerFixture, demo):
    mocker.patch.object(phases.sys, 'executable', 'C:\\Python\\python.exe')
    builder = Builder(_config(tmp_path, os_name='windows'), Console(quiet=True))
    builder._manifest = demo
    parts = builder._expand('{python} prune.py {build_dir}')
    assert parts == ['C:\\Python\\python.exe', 'prune.py', str(builder.config.cmake_dir)]


def test_tables_print_manifest_text_verbatim(tmp_path: Path, mocker: MockerFixture):
    """Brackets in a manifest's help text or an option value are printed, not parsed as markup."""
    builder = _builder(tmp_path, mocker)
    builder._manifest = parse_manifest(
        {
            'manifest_version': 1,
            'backend': {'name': 'demo', 'default_ref': 'v1'},
            'groups': {'g': {'help': 'Enables [experimental] and [/b] mode', 'cmake': ['G=ON']}},
            'options': {'o': {'help': 'O', 'default': '[red]x', 'cmake': 'O={value}'}},
        }
    )
    builder.groups = ['g']
    builder.option_values = {'o': '[red]x'}
    builder._print_selection()
    out = builder.console.export_text()
    assert 'Enables [experimental] and [/b] mode' in out
    assert '[red]x' in out


def test_manifest_before_source(tmp_path: Path):
    with pytest.raises(BuildError, match='only available after the source phase'):
        _ = Builder(_config(tmp_path), Console(quiet=True)).manifest


def test_run_executes_and_reports(tmp_path: Path, mocker: MockerFixture):
    builder = _builder(tmp_path, mocker)
    builder.env['DEMO'] = 'outer'
    cwd = tmp_path / 'made'
    check = 'import os, sys; sys.exit(os.environ["DEMO"] != "x")'
    builder.run([sys.executable, '-c', check], cwd=cwd, env={'DEMO': 'x'})
    assert cwd.is_dir()
    builder.run([sys.executable, '-c', 'raise SystemExit(3)'], fatal=False)
    with pytest.raises(BuildError, match='exit status 3'):
        builder.run([sys.executable, '-c', 'raise SystemExit(3)'])
    with pytest.raises(BuildError, match='Cannot run definitely-not-a-command'):
        builder.run(['definitely-not-a-command'])


def test_run_resolves_from_the_build_path(tmp_path: Path, mocker: MockerFixture):
    builder = _builder(tmp_path, mocker)
    assert builder.env['PATH'].startswith(tools.scripts_dir())
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    exe = bin_dir / 'demo-tool.exe'
    exe.write_text('')
    exe.chmod(0o755)
    builder.env['PATH'] = str(bin_dir)
    builder.config.verbose = True
    spy = mocker.patch.object(subprocess, 'run', return_value=subprocess.CompletedProcess([], 0))
    builder.run(['demo-tool.exe', '--flag'])
    assert spy.call_args.args[0] == [str(exe), '--flag']


def test_run_logged_filters_noise(tmp_path: Path, mocker: MockerFixture):
    builder = _builder(tmp_path, mocker)
    log = tmp_path / 'build.log'
    script = tmp_path / 'noisy.py'
    script.write_text(
        'print("ok line"); print("x.cxx(1): warning C4244: conversion"); '
        'print("x.cxx(2): error C2065: undeclared"); raise SystemExit(1)'
    )
    with pytest.raises(BuildError):
        builder.run([sys.executable, str(script)], log=log)
    out = builder.console.export_text()
    assert 'ok line' in out
    assert 'error C2065: undeclared' in out
    assert 'C4244' not in out
    assert 'C4244' in log.read_text()
    assert 'full log is in' in out
    builder.run([sys.executable, '-c', 'print("clean")'], log=log)
    assert log.read_text() == 'clean\n'


@pytest.mark.parametrize('verbose', [False, True])
def test_run_prints_the_environment_only_when_verbose(tmp_path: Path, verbose):
    console = Console(record=True, width=4000, force_terminal=False)
    builder = Builder(_config(tmp_path, verbose=verbose), console)
    builder.run(['cmake', '--build', 'b'], env={'ARCHFLAGS': '-arch arm64'})
    shown = "$ ARCHFLAGS='-arch arm64' cmake --build b" if verbose else '$ cmake --build b'
    assert console.export_text() == f'{shown}\n'


def test_existing_checkout_update_and_ref(tmp_path: Path, mocker: MockerFixture):
    src = tmp_path / 'src'
    (src / '.git').mkdir(parents=True)
    builder = _builder(tmp_path, mocker, source=src, update=True)
    git = mocker.patch.object(builder, 'git', return_value='')
    builder._existing_checkout(src)
    calls = [c.args for c in git.call_args_list]
    assert ('fetch', '--progress', '--tags', 'origin') in calls
    assert any('checkout' in c and f'origin/{DEFAULT_REFS["vtk"]}' in c for c in calls)
    builder = _builder(tmp_path, mocker, source=src, ref='main')
    git = mocker.patch.object(
        builder, 'git', side_effect=lambda *a, **_: a[-1] if a[0] == 'rev-parse' else ''
    )
    builder._existing_checkout(src)
    assert any('checkout' in c and 'main' in c for c in [c.args for c in git.call_args_list])
    git = mocker.patch.object(builder, 'git', side_effect=subprocess.CalledProcessError(128, 'g'))
    with pytest.raises(BuildError, match='git status failed'):
        builder._existing_checkout(src)
    git = mocker.patch.object(
        builder, 'git', side_effect=lambda *a, **_: a[-1] if a[0] == 'rev-parse' else ''
    )
    mocker.patch.object(builder, 'confirm', return_value=False)
    builder._existing_checkout(src)
    assert not any('checkout' in c.args for c in git.call_args_list)


def test_prompts_print_brackets_verbatim():
    assert phases.Prompt('Start? [y/n, or a number of jobs]').prompt.plain.startswith(
        'Start? [y/n, or a number of jobs]'
    )
    assert phases.Confirm('Use [bold]?').prompt.plain.startswith('Use [bold]?')


def test_tools_pip_paths(tmp_path: Path, mocker: MockerFixture):
    mocker.patch.object(
        tools.shutil,
        'which',
        side_effect=lambda name, **_: None if name == 'ninja' else f'/usr/bin/{name}',
    )
    mocker.patch.object(tools, 'executable_version', return_value='99.0')
    mocker.patch.object(tools, 'python_build_packages', return_value=[])
    builder = _builder(tmp_path, mocker)
    mocker.patch.object(tools, 'pip_install_command', return_value=None)
    with pytest.raises(BuildError, match='no pip or uv found'):
        builder.phase_tools()
    mocker.patch.object(tools, 'pip_install_command', return_value=['pip', 'install', 'ninja'])
    mocker.patch.object(phases.sys.stdin, 'isatty', return_value=False)
    with pytest.raises(BuildError, match='Pass --yes'):
        builder.phase_tools()
    builder.config.yes = True
    mocker.patch.object(builder, 'run')
    with pytest.raises(BuildError, match='Still not usable after install: ninja'):
        builder.phase_tools()
    builder = _builder(tmp_path, mocker, install=True, yes=True)
    mocker.patch.object(tools.shutil, 'which', side_effect=lambda name, **_: f'/usr/bin/{name}')
    mocker.patch.object(tools, 'pip_install_command', return_value=None)
    with pytest.raises(BuildError, match='--install needs pip or uv'):
        builder.phase_tools()


@pytest.mark.parametrize('repair', [True, False])
@pytest.mark.parametrize('install', [True, False])
@pytest.mark.parametrize('missing', [(), ('ninja', 'auditwheel')])
def test_install_list_follows_flags(
    tmp_path: Path, mocker: MockerFixture, repair, install, missing
):
    absent = set(missing)
    mocker.patch.object(
        tools.shutil,
        'which',
        side_effect=lambda name, **_: None if name in absent else f'/usr/bin/{name}',
    )
    mocker.patch.object(tools, 'executable_version', return_value='99.0')
    mocker.patch.object(tools, 'python_build_packages', return_value=[])
    mocker.patch.object(
        tools, 'pip_install_command', side_effect=lambda *s, **_: ['pip', 'install', *s]
    )
    builder = _builder(tmp_path, mocker, repair=repair, install=install, yes=True)
    run = mocker.patch.object(builder, 'run', side_effect=lambda *_a, **_k: absent.clear())
    builder.phase_tools()
    bullets = [
        line.removeprefix('  • ')
        for line in builder.console.export_text().splitlines()
        if line.startswith('  • ')
    ]
    specs = (['ninja>=1.11'] + (['auditwheel'] if repair else [])) if missing else []
    expected = [f'{spec} into {sys.prefix}' for spec in specs]
    if install:
        expected.append(f'the vtk wheel into {sys.prefix}, once it is built')
    assert bullets == expected
    assert run.called == bool(specs)
    if specs:
        assert run.call_args.args[0] == ['pip', 'install', *specs]


@pytest.mark.parametrize(
    ('kwargs', 'tty', 'answer', 'error'),
    [
        ({'yes': True}, False, None, None),
        ({'dry_run': True}, False, None, None),
        ({}, False, None, 'Pass --yes'),
        ({}, True, True, None),
        ({}, True, False, 'Nothing was installed'),
        ({'interactive': True}, True, True, None),
        ({'interactive': True, 'yes': True}, True, None, None),
    ],
)
def test_confirm_installs(tmp_path: Path, mocker: MockerFixture, kwargs, tty, answer, error):
    mocker.patch.object(phases.sys.stdin, 'isatty', return_value=tty)
    ask = mocker.patch.object(phases.Confirm, 'ask', return_value=answer)
    console = Console(record=True, width=4000, force_terminal=False)
    builder = Builder(_config(tmp_path, **{'dry_run': False, **kwargs}), console)
    builder.confirm_installs([])
    assert console.export_text() == ''
    if error:
        with pytest.raises(BuildError, match=error):
            builder.confirm_installs(['ninja into /env'])
    else:
        builder.confirm_installs(['ninja into /env'])
    assert ask.call_count == (answer is not None)
    assert console.export_text() == 'The build installs:\n  • ninja into /env\n'


def test_git_version_suffix(tmp_path: Path, mocker: MockerFixture):
    cfg = _config(tmp_path, dry_run=False)
    (cfg.src_dir / '.git').mkdir(parents=True)
    builder = Builder(cfg, Console(quiet=True))
    builder._manifest = find_manifest('vtk', None)
    mocker.patch.object(builder, 'git', return_value='v9.7.0')
    assert builder.version_suffix() is None
    err = subprocess.CalledProcessError(128, ['git'])

    def describe_fails(*args, **_):
        if 'describe' in args:
            raise err
        return 'abcdef12'

    mocker.patch.object(builder, 'git', side_effect=describe_fails)
    assert builder.version_suffix() == 'dev0+gabcdef12'
    mocker.patch.object(builder, 'git', side_effect=err)
    assert builder.version_suffix() is None


def test_confirm_asks_on_a_terminal(tmp_path: Path, mocker: MockerFixture):
    mocker.patch.object(phases.sys.stdin, 'isatty', return_value=True)
    ask = mocker.patch.object(phases.Confirm, 'ask', return_value=False)
    console = Console(record=True, width=4000, force_terminal=False)
    builder = Builder(_config(tmp_path, dry_run=False, interactive=True), console)
    assert builder.confirm('Go?') is False
    assert ask.call_args.args[0] == 'Go?'
    builder = Builder(_config(tmp_path, dry_run=False), console)
    assert builder.confirm('Go?') is True
    assert console.export_text() == ''
    builder = Builder(_config(tmp_path, interactive=True), console)
    assert builder.confirm('Go?') is True
    assert 'Go? -> yes (dry run)' in console.export_text()
    mocker.patch.object(phases.sys.stdin, 'isatty', return_value=False)
    builder = Builder(_config(tmp_path, interactive=True, dry_run=False), console)
    assert builder.confirm('Go [now]?', default=False) is False
    assert 'Go [now]? -> no (no terminal)' in console.export_text()


@pytest.mark.usefixtures('no_tools')
def test_clone_declined(tmp_path: Path, mocker: MockerFixture):
    builder = Builder(_config(tmp_path, dry_run=False), Console(quiet=True))
    mocker.patch.object(builder, 'confirm', return_value=False)
    with pytest.raises(BuildError, match='Nothing to build without a source checkout'):
        builder.phase_source()


def test_groups_phase_errors_and_required_display(tmp_path: Path, mocker: MockerFixture, demo):
    builder = _builder(tmp_path, mocker, with_=['nope'])
    with pytest.raises(BuildError, match='Unknown group'):
        builder.phase_groups()
    builder = _builder(tmp_path, mocker)
    builder._manifest = demo
    builder.groups = resolve_groups(demo, 'linux')
    builder.option_values = resolve_options(demo, None)
    builder._print_selection()
    assert '(required)' in builder.console.export_text()


def test_version_command_output(tmp_path: Path, mocker: MockerFixture):
    src = tmp_path / 'src'
    (src / '.git').mkdir(parents=True)
    (src / 'pyvista-build.toml').write_text(
        'manifest_version = 1\n'
        '[backend]\nname = "demo"\ndefault_ref = "v1"\n'
        'version_command = "{python} version.py"\n'
    )
    builder = Builder(
        _config(tmp_path, backend='demo', source=src, dry_run=False), Console(quiet=True)
    )
    builder._manifest = find_manifest('demo', src)
    (src / 'version.py').write_text('print("post5")')
    assert builder.version_suffix() == 'post5'
    (src / 'version.py').write_text('')
    assert builder.version_suffix() is None
    mocker.patch.object(subprocess, 'run', side_effect=FileNotFoundError('gone'))
    with pytest.raises(BuildError, match='Cannot run version_command'):
        builder.version_suffix()


def test_paths_link_only_on_a_real_terminal(tmp_path: Path):
    """Forced color into a pipe, as in a docs build, gets colors but no hyperlinks."""

    class Terminal(StringIO):
        def isatty(self) -> bool:
            return True

    for file, linked in [(StringIO(), False), (Terminal(), True)]:
        console = Console(file=file, force_terminal=True, width=200)
        Builder(_config(tmp_path), console).say('Wheel: /a/b')
        assert '\x1b[' in file.getvalue()
        assert ('\x1b]8;' in file.getvalue()) == (linked and os.name != 'nt')
    closed = StringIO()
    closed.close()
    assert not Builder(_config(tmp_path), Console(file=closed, force_terminal=True))._links


def test_narration_highlighter_edge_cases(mocker: MockerFixture):
    """Paths stay one span, ``~`` alone is not a path, and an unlinkable path is still colored."""
    text = phases.Text('/a/2026-10/b about ~5 minutes')
    phases._HIGHLIGHT.highlight(text)
    spans = [(text.plain[s.start : s.end], str(s.style)) for s in text.spans]
    assert spans == [('/a/2026-10/b', _path_style('/a/2026-10/b')), ('5', 'cyan')]
    mocker.patch.object(
        phases,
        'Path',
        return_value=mocker.Mock(
            **{
                'expanduser.return_value.as_uri.side_effect': ValueError,
            }
        ),
    )
    text = phases.Text('flags /MP')
    phases._HIGHLIGHT.highlight(text)
    assert [str(s.style) for s in text.spans] == ['magenta']


def test_narration_highlighter():
    """URLs, paths, flags and versions get a style, other words do not."""
    text = phases.Text(
        'clone https://x.org/vtk.git at v9.7.1 into /a/b, --repair skipped, 15 jobs'
    )
    phases._HIGHLIGHT.highlight(text)
    styled = {text.plain[s.start : s.end]: str(s.style) for s in text.spans}
    assert styled.pop('/a/b') == _path_style('/a/b')
    assert styled == {
        'https://x.org/vtk.git': 'underline blue',
        '--repair': 'bold cyan',
        'skipped': 'yellow',
        'v9.7.1': 'cyan',
        '15': 'cyan',
    }


@pytest.mark.skipif(shutil.which('git') is None, reason='needs git')
def test_update_follows_the_remote_branch(
    tmp_path: Path, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
):
    """``--update`` on a branch ref checks out the fetched commit, not the stale local branch."""
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', os.devnull)
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')

    def git(*args, cwd):
        return subprocess.run(
            [str(shutil.which('git')), '-c', 'user.name=t', '-c', 'user.email=t@t', *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    origin = tmp_path / 'origin'
    origin.mkdir()
    git('init', '-q', '-b', 'main', cwd=origin)
    git('commit', '-q', '--allow-empty', '-m', 'one', cwd=origin)
    git('clone', '-q', str(origin), str(tmp_path / 'src'), cwd=tmp_path)
    git('commit', '-q', '--allow-empty', '-m', 'two', cwd=origin)
    builder = _builder(tmp_path, mocker, source=tmp_path / 'src', update=True, ref='main')
    builder._existing_checkout(tmp_path / 'src')
    assert git('rev-parse', 'HEAD', cwd=tmp_path / 'src') == git('rev-parse', 'HEAD', cwd=origin)
    (tmp_path / 'src' / 'notes.txt').write_text('mine')
    git('tag', 'v1', 'HEAD~1', cwd=origin)
    builder.config.ref = 'v1'
    builder._existing_checkout(tmp_path / 'src')
    tagged = git('rev-parse', 'v1^{commit}', cwd=origin)
    assert git('rev-parse', 'HEAD', cwd=tmp_path / 'src') == tagged


def test_recheck_tools_after_clone(tmp_path: Path, mocker: MockerFixture):
    """Tool versions only the cloned manifest asks for are checked after the clone."""
    builder = _builder(tmp_path, mocker)
    shipped = builder.manifest
    run = mocker.patch.object(builder, 'run')
    confirm = mocker.patch.object(builder, 'confirm_installs')
    builder.recheck_tools()
    assert not run.called
    builder._checked_tools = dict(shipped.tools)
    builder.recheck_tools()
    assert not run.called

    builder._manifest = replace(shipped, tools={**shipped.tools, 'cmake': '>=99'})
    detect = mocker.patch.object(tools, 'detect', return_value=True)
    builder.recheck_tools()
    assert "The checkout's manifest asks for cmake>=99" in builder.console.export_text()
    assert not run.called
    assert builder._checked_tools['cmake'] == '>=99'

    detect.reset_mock()
    builder._checked_tools = {}
    builder._manifest = replace(shipped, tools={'cmake': '', 'ninja': ''})
    builder.recheck_tools()
    assert not detect.called

    builder._checked_tools = {}
    builder._manifest = replace(shipped, tools={'cmake': '>=99'})
    detect.return_value = False
    mocker.patch.object(tools, 'pip_install_command', return_value=None)
    with pytest.raises(BuildError, match='no pip or uv found'):
        builder.recheck_tools()
    builder._checked_tools = {}
    mocker.patch.object(tools, 'pip_install_command', return_value=['pip', 'install'])
    with pytest.raises(BuildError, match='Still not usable after install: cmake>=99'):
        builder.recheck_tools()
    assert confirm.call_args.args[0] == [f'cmake>=99 into {sys.prefix}']
    builder._checked_tools = {}
    detect.side_effect = [False, True]
    builder.recheck_tools()
    assert run.call_args.args[0] == ['pip', 'install']
    builder._checked_tools = {}
    builder.config.dry_run = True
    detect.side_effect = None
    builder.recheck_tools()
    assert run.call_count == 3


def test_patchelf_offered_on_linux(tmp_path: Path, mocker: MockerFixture):
    """auditwheel needs patchelf, so a Linux build that repairs offers to install it."""
    absent = {'patchelf'}
    mocker.patch.object(
        tools.shutil,
        'which',
        side_effect=lambda name, **_: None if name in absent else f'/usr/bin/{name}',
    )
    mocker.patch.object(tools, 'executable_version', return_value='99.0')
    mocker.patch.object(tools, 'python_build_packages', return_value=[])
    mocker.patch.object(
        tools, 'pip_install_command', side_effect=lambda *s, **_: ['pip', 'install', *s]
    )
    builder = _builder(tmp_path, mocker, repair=True, yes=True)
    run = mocker.patch.object(builder, 'run', side_effect=lambda *_a, **_k: absent.clear())
    builder.phase_tools()
    assert run.call_args.args[0] == ['pip', 'install', 'patchelf']
    assert f'  • patchelf into {sys.prefix}' in builder.console.export_text()


@pytest.mark.parametrize(
    ('line', 'noisy'),
    [
        ('x.cxx(1): warning C4244: conversion', True),
        ('x.h:3:5: note: declared here', True),
        ('    with', True),
        ('  [', True),
        ('  ]', True),
        ('    _Ty=double', True),
        ('x.cxx(2): error C2065: undeclared', False),
        ('x.cxx:4:1: error: expected ;', False),
    ],
)
def test_noisy_build_lines(line, noisy):
    """Compiler notes and template-context lines are filtered, errors are kept."""
    assert bool(phases._NOISY_BUILD_LINE.search(line)) is noisy


def test_macos_compiler_needs_command_line_tools(tmp_path: Path, mocker: MockerFixture):
    """On macOS the ``c++`` shim without the Command Line Tools counts as no compiler."""
    mocker.patch.object(tools.shutil, 'which', side_effect=lambda name, **_: f'/usr/bin/{name}')
    mocker.patch.object(tools, 'executable_version', return_value='99.0')
    mocker.patch.object(tools, 'python_build_packages', return_value=[])
    run = mocker.patch.object(
        tools.subprocess, 'run', return_value=subprocess.CompletedProcess([], 2)
    )
    compiler = tools.required_tools('macos', {}, repair=False)[-1]
    assert not tools.detect(compiler)
    assert compiler.path is None
    assert run.call_args.args[0] == ['xcode-select', '-p']
    with pytest.raises(BuildError, match='Install the Xcode Command Line Tools'):
        _builder(tmp_path, mocker, os_name='macos').phase_tools()
    run.return_value = subprocess.CompletedProcess([], 0)
    assert tools.detect(compiler)
    run.side_effect = FileNotFoundError
    assert not tools.detect(compiler)
    run.reset_mock()
    assert tools.detect(tools.required_tools('linux', {}, repair=False)[-1])
    assert not run.called
