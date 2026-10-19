"""Tests for the backend build engine."""

from __future__ import annotations

from dataclasses import replace
from io import StringIO
import json
import os
import re
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING
import zipfile

import pytest
from rich.console import Console

from pyvista._build import BuildConfig
from pyvista._build import Builder
from pyvista._build import BuildError
from pyvista._build import phases
from pyvista._build import tools
from pyvista._build.manifest import OS_NAMES
from pyvista._build.manifest import cmake_unset
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


@pytest.mark.usefixtures('no_tools')
@pytest.mark.parametrize('os_name', OS_NAMES)
def test_dry_run_commands(tmp_path: Path, os_name):
    console = Console(record=True, width=4000, force_terminal=False)
    cfg = _config(
        tmp_path,
        os_name=os_name,
        defines=['FOO=1', 'VTK_WRAP_PYTHON=OFF'],
        ref='v9.7.0',
        repair=True,
        verbose=True,
    )
    builder = Builder(cfg, console)
    assert builder.build() is None
    out = console.export_text()
    assert out.startswith('Building backend: vtk')
    assert 'git clone --progress --filter=blob:none https://gitlab.kitware.com/vtk/vtk.git' in out
    assert 'checkout v9.7.0' in out
    assert '-DFOO=1' in out
    assert '-DVTK_VERSION_SUFFIX= ' in out
    configure = next(line for line in out.splitlines() if ' -GNinja' in line)
    assert configure.index('-DVTK_WRAP_PYTHON=OFF') > configure.index('-DVTK_WRAP_PYTHON=ON')
    assert ('_PYTHON_HOST_PLATFORM=macosx-11.0-' in configure) == (os_name == 'macos')
    repairer = {
        'linux': 'auditwheel repair',
        'macos': 'delocate-wheel',
        'windows': 'delvewheel repair',
    }
    assert repairer[os_name] in out
    assert 'setup.py bdist_wheel' in out
    hint = next(line for line in out.splitlines() if '--force-reinstall' in line)
    assert str(cfg.wheelhouse) in hint
    assert 'COMPILER_LAUNCHER' not in out


@pytest.mark.usefixtures('no_tools')
def test_dry_run_flags(tmp_path: Path):
    console = Console(record=True, width=4000, force_terminal=False)
    cfg = _config(
        tmp_path,
        minimal=True,
        build_type='Debug',
        jobs=3,
        ref='main',
        version_suffix='',
        options=['smp=TBB'],
        repair=False,
    )
    Builder(cfg, console).build()
    out = console.export_text()
    assert 'COMPILER_LAUNCHER' not in out
    assert '-DCMAKE_BUILD_TYPE=Debug' in out
    assert '--parallel 3' in out
    assert '-DVTK_GROUP_ENABLE_Rendering=NO' in out
    assert '-DVTK_SMP_IMPLEMENTATION_TYPE=TBB' in out
    assert '-DVTK_VERSION_SUFFIX= ' in out
    assert 'Skipped (--no-repair).' in out
    hint = next(line for line in out.splitlines() if '--force-reinstall' in line)
    assert str(cfg.dist_dir) in hint


@pytest.mark.usefixtures('no_tools')
def test_dry_run_dev_suffix(tmp_path: Path):
    console = Console(record=True, width=4000, force_terminal=False)
    Builder(_config(tmp_path, ref='main'), console).build()
    assert '-DVTK_VERSION_SUFFIX=dev0+g<sha>' in console.export_text()


def test_unknown_backend(tmp_path: Path):
    with pytest.raises(BuildError, match='Unknown backend'):
        Builder(_config(tmp_path, backend='nope'), Console(quiet=True)).build()


def test_missing_git(tmp_path: Path, mocker: MockerFixture):
    """A missing git stops at the tools step, before anything is cloned."""
    mocker.patch.object(tools.shutil, 'which', return_value=None)
    builder = Builder(_config(tmp_path), Console(quiet=True))
    source = mocker.spy(builder, 'phase_source')
    with pytest.raises(BuildError, match='git: Install git'):
        builder.build()
    assert not source.called
    with pytest.raises(BuildError, match='git was not found'):
        builder.phase_source()


def test_blocking_tool(tmp_path: Path, mocker: MockerFixture):
    mocker.patch.object(
        tools.shutil,
        'which',
        side_effect=lambda name, **_: (
            None if name in ('c++', 'clang++', 'g++') else f'/usr/bin/{name}'
        ),
    )
    mocker.patch.object(tools, 'executable_version', return_value='99.0')
    with pytest.raises(BuildError, match='build-essential'):
        Builder(_config(tmp_path), Console(quiet=True)).build()


def test_pip_tools_offered(tmp_path: Path, mocker: MockerFixture):
    missing = {'cmake', 'auditwheel'}
    mocker.patch.object(
        tools.shutil,
        'which',
        side_effect=lambda name, **_: None if name in missing else f'/usr/bin/{name}',
    )
    mocker.patch.object(tools, 'executable_version', return_value='99.0')
    mocker.patch.object(tools, 'python_build_packages', return_value=['wheel'])
    mocker.patch.object(
        tools, 'pip_install_command', side_effect=lambda *s, args=(): ['pip', 'install', *args, *s]
    )
    console = Console(record=True, width=4000, force_terminal=False)
    Builder(_config(tmp_path, repair=True), console).build()
    out = console.export_text()
    bullets = [f'  • {spec} into {sys.prefix}' for spec in ('cmake>=3.22', 'auditwheel', 'wheel')]
    assert '\n'.join(['The build installs:', *bullets]) in out
    assert re.search(r'wheel \(python\)\s+│ -\s+│ missing', out)
    assert re.search(r'setuptools \(python\)\s+│ .*│ ok', out)
    assert f'$ {phases._join(["pip", "install", "cmake>=3.22", "auditwheel", "wheel"])}' in out


def test_too_old_tool(tmp_path: Path, mocker: MockerFixture):
    mocker.patch.object(tools.shutil, 'which', side_effect=lambda name, **_: f'/usr/bin/{name}')
    mocker.patch.object(tools, 'executable_version', return_value='3.10')
    mocker.patch.object(tools, 'python_build_packages', return_value=[])
    mocker.patch.object(
        tools, 'pip_install_command', side_effect=lambda *s, **_: ['pip', 'install', *s]
    )
    console = Console(record=True, width=4000, force_terminal=False)
    Builder(_config(tmp_path), console).build()
    out = console.export_text()
    assert 'wrong version' in out
    assert f'$ {phases._join(["pip", "install", "cmake>=3.22"])}' in out


@pytest.mark.usefixtures('no_tools')
def test_hooks_and_init_cache(tmp_path: Path):
    src = tmp_path / 'src'
    (src / '.git').mkdir(parents=True)
    (src / 'pyvista-build.toml').write_text(
        'manifest_version = 1\n'
        '[backend]\nname = "demo"\ndefault_ref = "v1"\n'
        '[cmake]\ninit_cache = { linux = "ci/linux.cmake" }\n'
        '[hooks]\npost_build = ["{python} prune.py {build_dir}"]\n'
    )
    console = Console(record=True, width=4000, force_terminal=False)
    Builder(_config(tmp_path, backend='demo', source=src, version_suffix=''), console).build()
    out = console.export_text()
    assert f'-C {src / "ci" / "linux.cmake"}' in out
    assert f'prune.py {tmp_path / "vtk-build" / "build"} (in {src})' in out
    assert 'git clone' not in out


def _fake_git(head: str, status: str = ' M file.cxx'):
    """Return a ``subprocess.run`` stand-in for a checkout at ``head`` with ``status`` output."""

    def run(cmd, **_):
        out = {'status': status, 'rev-parse': head if cmd[-1] == 'HEAD' else 'b' * 40}
        return subprocess.CompletedProcess(cmd, 0, stdout=out.get(cmd[1], ''), stderr='')

    return run


def test_existing_checkout_dirty(tmp_path: Path, mocker: MockerFixture):
    src = tmp_path / 'vtk-build' / 'src'
    (src / '.git').mkdir(parents=True)
    mocker.patch.object(tools.shutil, 'which', side_effect=lambda name, **_: f'/usr/bin/{name}')
    mocker.patch.object(subprocess, 'run', side_effect=_fake_git('a' * 40))
    with pytest.raises(BuildError, match='local changes'):
        Builder(_config(tmp_path, ref='v9.7.0'), Console(quiet=True)).build()


@pytest.mark.usefixtures('no_tools')
def test_existing_checkout_at_ref_keeps_edits(tmp_path: Path, mocker: MockerFixture):
    src = tmp_path / 'vtk-build' / 'src'
    (src / '.git').mkdir(parents=True)
    run = mocker.patch.object(subprocess, 'run', side_effect=_fake_git('b' * 40))
    console = Console(record=True, width=4000, force_terminal=False)
    Builder(_config(tmp_path, ref='v9.7.0'), console).build()
    assert not any(c.args[0][1] == 'status' for c in run.call_args_list)
    assert 'checkout v9.7.0' not in console.export_text()


def test_state_change_prompts_reconfigure(tmp_path: Path, mocker: MockerFixture):
    cmake_dir = tmp_path / 'vtk-build' / 'build'
    cmake_dir.mkdir(parents=True)
    (cmake_dir / 'CMakeCache.txt').write_text('')
    (cmake_dir / '.pyvista-build.json').write_text('{"groups": []}')
    src = tmp_path / 'vtk-build' / 'src'
    (src / '.git').mkdir(parents=True)
    mocker.patch.object(tools.shutil, 'which', side_effect=lambda name, **_: f'/usr/bin/{name}')
    mocker.patch.object(tools, 'executable_version', return_value='99.0')
    mocker.patch.object(tools, 'python_build_packages', return_value=[])
    console = Console(record=True, width=4000, force_terminal=False)
    Builder(_config(tmp_path, version_suffix=''), console).build()
    assert 'differs from the one this tree was set up with' in console.export_text()


def test_env_options_and_version_command(tmp_path: Path, mocker: MockerFixture):
    src = tmp_path / 'src'
    (src / '.git').mkdir(parents=True)
    (src / 'pyvista-build.toml').write_text(
        'manifest_version = 1\n'
        '[backend]\nname = "demo"\ndefault_ref = "v1"\n'
        'version_command = "{python} version.py"\n'
        '[options.lto]\nhelp = "LTO"\ndefault = false\nenv = "DEMO_LTO"\n'
        '[options.abi3]\nhelp = "abi3"\ndefault = true\ncmake = "DEMO_ABI3={value}"\n'
        '[hooks]\npost_configure = ["{python} check.py"]\n'
    )
    mocker.patch.object(tools.shutil, 'which', side_effect=lambda name, **_: f'/usr/bin/{name}')
    mocker.patch.object(tools, 'executable_version', return_value='99.0')
    mocker.patch.object(tools, 'python_build_packages', return_value=[])
    console = Console(record=True, width=4000, force_terminal=False)
    cfg = _config(tmp_path, backend='demo', source=src, options=['lto=1'], verbose=True)
    Builder(cfg, console).build()
    out = console.export_text()
    assert '$ DEMO_LTO=1 cmake' in out
    assert re.search(r'\$ DEMO_LTO=1 \S+ check\.py', out)
    assert '-DDEMO_ABI3=ON' in out
    assert '-DDEMO_LTO' not in out
    assert '-DVTK_VERSION_SUFFIX=<version_command>' in out


@pytest.mark.usefixtures('no_tools')
def test_dry_run_needs_checkout_without_shipped_manifest(tmp_path: Path):
    cfg = _config(tmp_path, backend='demo', url='https://example.invalid/demo.git')
    console = Console(record=True, width=4000, force_terminal=False)
    with pytest.raises(BuildError, match=re.escape('Pass --source <checkout> to dry-run it')):
        Builder(cfg, console).build()
    assert '1/8 Tools' not in console.export_text()


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


def test_msvc_environment(mocker: MockerFixture, tmp_path: Path):
    run = mocker.patch.object(
        tools.subprocess,
        'run',
        return_value=subprocess.CompletedProcess(
            '', 0, stdout='INCLUDE=C:\\inc\nPATH=C:\\vc;old\nUSER=Jürgen\n'.encode('utf-16-le')
        ),
    )
    vcvarsall = tmp_path / 'vcvarsall.bat'
    before = {'PATH': 'old', 'KEEP': '1'}
    env = tools.msvc_environment(vcvarsall, env=before)
    assert env == {'PATH': 'C:\\vc;old', 'KEEP': '1', 'INCLUDE': 'C:\\inc', 'USER': 'Jürgen'}
    assert before == {'PATH': 'old', 'KEEP': '1'}
    assert run.call_args.args[0] == f'cmd /u /c ""{vcvarsall}" x64 >nul && set"'


def test_msvc_failure_is_a_build_error(tmp_path: Path, mocker: MockerFixture):
    mocker.patch.object(tools.shutil, 'which', return_value=None)
    mocker.patch.object(tools, 'find_vcvarsall', return_value=tmp_path / 'vcvarsall.bat')
    mocker.patch.object(
        tools, 'msvc_environment', side_effect=subprocess.CalledProcessError(2, 'cmd')
    )
    builder = _builder(tmp_path, mocker, os_name='windows')
    with pytest.raises(BuildError, match=r'vcvarsall\.bat failed with exit status 2'):
        builder._activate_msvc(tools.required_tools('windows', {}, repair=False))


def _builder(tmp_path: Path, mocker: MockerFixture, **kwargs) -> Builder:
    """Return a builder on the shipped VTK manifest with the source phase done."""
    cfg = _config(tmp_path, dry_run=False, **kwargs)
    builder = Builder(cfg, Console(record=True, width=4000, force_terminal=False))
    builder._manifest = find_manifest('vtk', None)
    builder.groups = resolve_groups(builder.manifest, cfg.os_name)
    builder.option_values = resolve_options(builder.manifest, None)
    mocker.patch.object(builder, 'version_suffix', return_value=None)
    return builder


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
    crlf = (
        'import sys; sys.stdout.buffer.write(b"one\\r\\n"); sys.stderr.buffer.write(b"two\\r\\n")'
    )
    builder.run([sys.executable, '-c', crlf], log=log)
    assert sorted(log.read_bytes().decode().splitlines(keepends=True)) == [
        f'one{os.linesep}',
        f'two{os.linesep}',
    ]
    out = builder.console.export_text()
    assert 'two\n' in out
    assert '\r' not in out


def test_run_folds_output_into_a_status_line(tmp_path: Path, mocker: MockerFixture):
    """Standard output becomes one status line, stderr still prints, a failure shows the tail."""
    builder = _builder(tmp_path, mocker)
    builder.console = Console(record=True, width=4000, force_terminal=True)
    status = mocker.patch.object(builder.console, 'status')
    mocker.patch.object(phases, '_TICK_SECONDS', 0.01)
    script = tmp_path / 'chatty.py'
    script.write_text(
        'import sys, time\n'
        'time.sleep(0.2)\n'
        'sys.stderr.write("Updating files: 50%\\rUpdating files: 99%\\r")\n'
        'sys.stdout.write("Receiving 10%\\r")\n'
        'for i in range(50): print(f"step {i}")\n'
        'print("careful", file=sys.stderr)\n'
        'raise SystemExit(int(sys.argv[1]))\n'
    )
    builder.run([sys.executable, str(script), '0'])
    out = builder.console.export_text()
    assert 'careful' in out
    assert 'step 49' not in out
    assert 'Updating files' not in out
    assert re.search(r'Done in \d+s\n', out)
    update = status.return_value.__enter__.return_value.update
    shown = [c.args[0].split(' [dim]')[0] for c in update.call_args_list]
    assert shown[0] == phases._short_name([sys.executable])
    assert shown.count(shown[0]) > 1
    assert [s for s in shown if s.startswith('step')][-1] == 'step 49'
    assert {'Updating files: 99%', 'Receiving 10%'} <= set(shown)
    assert re.search(r' \[dim\]\d+s\[/dim\]$', update.call_args.args[0])
    with pytest.raises(BuildError, match='exit status 2'):
        builder.run([sys.executable, str(script), '2'])
    out = builder.console.export_text()
    assert 'step 49' in out
    assert 'step 9\n' not in out

    builder.config.verbose = True
    builder.run([sys.executable, str(script), '0'], log=tmp_path / 'build.log')
    assert 'step 0' in builder.console.export_text()
    assert 'careful' in (tmp_path / 'build.log').read_text()
    assert 'Updating files' not in (tmp_path / 'build.log').read_text()
    with pytest.raises(BuildError):
        builder.run([sys.executable, str(script), '1'], log=tmp_path / 'build.log')
    assert builder.console.export_text().count('step 49') == 1


def test_short_name():
    assert phases._short_name(['git', 'clone', 'https://x/y.git']) == 'git clone'
    assert phases._short_name(['/usr/bin/cmake', '-S', 'src']) == 'cmake'
    assert phases._short_name(['python', 'setup.py', 'bdist_wheel']) == 'python setup.py'
    assert phases._short_name(['python', '/a/b.py']) == 'python'


@pytest.mark.parametrize(('seconds', 'text'), [(4.6, '5s'), (64, '1m 04s'), (3725, '1h 02m')])
def test_duration(seconds, text):
    assert phases._duration(seconds) == text


@pytest.mark.skipif(sys.platform == 'win32', reason='POSIX sessions and signals')
def test_run_captured_in_own_session(tmp_path: Path, mocker: MockerFixture):
    """A folded command leaves the terminal alone, and Ctrl-C still reaches it."""
    builder = _builder(tmp_path, mocker)
    builder.console = Console(record=True, width=4000, force_terminal=True)
    builder.run([sys.executable, '-c', 'import os; print(os.getsid(0))'], log=tmp_path / 'a.log')
    assert int((tmp_path / 'a.log').read_text()) != os.getsid(0)

    status = mocker.patch.object(builder.console, 'status')

    def press_ctrl_c(text: str) -> None:
        if text.startswith('ready'):
            raise KeyboardInterrupt

    status.return_value.__enter__.return_value.update.side_effect = press_ctrl_c
    mocker.patch.object(phases, '_TICK_SECONDS', 60)
    marker = tmp_path / 'interrupted'
    script = tmp_path / 'wait.py'
    script.write_text(
        'import signal, sys, time\n'
        f'signal.signal(signal.SIGINT, lambda *_: open({str(marker)!r}, "w") and sys.exit(1))\n'
        'print("ready", flush=True)\n'
        'time.sleep(30)\n'
    )
    with pytest.raises(KeyboardInterrupt):
        builder.run([sys.executable, str(script)])
    assert marker.exists()


@pytest.mark.parametrize('verbose', [False, True])
def test_run_prints_the_environment_only_when_verbose(tmp_path: Path, verbose):
    console = Console(record=True, width=4000, force_terminal=False)
    builder = Builder(_config(tmp_path, verbose=verbose), console)
    builder.run(['cmake', '--build', 'b'], env={'ARCHFLAGS': '-arch arm64'})
    shown = "$ ARCHFLAGS='-arch arm64' cmake --build b" if verbose else '$ cmake --build b'
    assert console.export_text() == f'{shown}\n'


def test_run_passthrough(tmp_path: Path, mocker: MockerFixture):
    builder = _builder(tmp_path, mocker)
    spy = mocker.patch.object(subprocess, 'run', return_value=subprocess.CompletedProcess([], 0))
    builder.run([sys.executable, '-V'], passthrough=True)
    assert spy.called


def test_source_not_a_checkout(tmp_path: Path, mocker: MockerFixture):
    builder = _builder(tmp_path, mocker, source=tmp_path / 'plain', yes=True)
    (tmp_path / 'plain').mkdir()
    first = mocker.patch.object(builder, 'phase_tools')
    with pytest.raises(BuildError, match='not a git checkout'):
        builder.build()
    first.assert_not_called()


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


def test_windows_msvc(tmp_path: Path, mocker: MockerFixture):
    present = {'git', 'cmake', 'ninja', 'delvewheel'}

    def which(name, path=None, **_):
        if name == 'cl':
            return 'C:/vs/cl' if path and 'C:/vs' in path else None
        return f'C:/t/{name}' if name in present else None

    mocker.patch.object(tools.shutil, 'which', side_effect=which)
    mocker.patch.object(tools, 'executable_version', return_value='99.0')
    mocker.patch.object(tools, 'python_build_packages', return_value=[])
    mocker.patch.object(tools, 'find_vcvarsall', return_value=None)
    builder = _builder(tmp_path, mocker, os_name='windows')
    with pytest.raises(BuildError, match='msvc'):
        builder.phase_tools()
    vcvarsall = tmp_path / 'vcvarsall.bat'
    mocker.patch.object(tools, 'find_vcvarsall', return_value=vcvarsall)
    activated = mocker.patch.object(
        tools, 'msvc_environment', return_value={'PATH': 'C:/vs;C:/t', 'INCLUDE': 'x'}
    )
    mocker.patch.object(tools, 'msvc_arch', return_value='arm64')
    builder = _builder(tmp_path, mocker, os_name='windows')
    builder.phase_tools()
    activated.assert_called_once()
    assert activated.call_args.args == (vcvarsall, 'arm64')
    assert tools.find_vcvarsall.call_args.args == ('arm64',)
    assert builder.env['INCLUDE'] == 'x'


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


def test_configure_paths(tmp_path: Path, mocker: MockerFixture):
    builder = _builder(tmp_path, mocker, clean=True)
    cfg = builder.config
    cfg.cmake_dir.mkdir(parents=True)
    (cfg.cmake_dir / 'stale').write_text('')
    run = mocker.patch.object(builder, 'run')
    builder.phase_configure()
    assert not (cfg.cmake_dir / 'stale').exists()
    assert run.called
    assert (cfg.cmake_dir / phases.STATE_FILENAME).is_file()
    (cfg.cmake_dir / 'CMakeCache.txt').write_text('')
    builder = _builder(tmp_path, mocker)
    run = mocker.patch.object(builder, 'run')
    builder.phase_configure()
    assert not run.called
    assert 'already configured' in builder.console.export_text()
    builder = _builder(tmp_path, mocker, build_type='Debug')
    mocker.patch.object(
        builder, 'confirm', side_effect=lambda q, **_: not q.startswith('Reconfigure')
    )
    with pytest.raises(BuildError, match='keeps its earlier configuration'):
        builder.phase_configure()
    builder = _builder(tmp_path, mocker, reconfigure=True)
    mocker.patch.object(
        builder, 'confirm', side_effect=lambda q, **_: not q.startswith('Run cmake')
    )
    with pytest.raises(BuildError, match='Stopped before configure'):
        builder.phase_configure()


def test_configure_state(tmp_path: Path, mocker: MockerFixture):
    builder = _builder(tmp_path, mocker)
    cfg = builder.config
    run = mocker.patch.object(builder, 'run')
    builder.phase_configure()
    state_file = cfg.cmake_dir / phases.STATE_FILENAME
    stored = json.loads(state_file.read_text())
    assert stored == {'command': builder.configure_command(), 'env': {}}
    assert run.call_args.args[0] == stored['command']
    (cfg.cmake_dir / 'CMakeCache.txt').write_text('')
    mocker.patch.object(builder, 'version_suffix', return_value='dev0+gabcdef12')
    confirm = mocker.patch.object(builder, 'confirm', return_value=True)
    builder.phase_configure()
    assert 'new commit' in builder.console.export_text()
    assert [c.args[0] for c in confirm.call_args_list] == ['Run cmake?']
    assert '-DVTK_VERSION_SUFFIX=dev0+gabcdef12' in run.call_args.args[0]
    state_file.write_text('{')
    builder.phase_configure()
    assert 'no record of how this tree was configured' in builder.console.export_text()
    assert confirm.call_args_list[-2].args[0] == 'Reconfigure?'
    state_file.write_text(json.dumps({'command': ['cmake'], 'env': {}}))
    confirm.reset_mock()
    builder.phase_configure()
    assert confirm.call_args_list[0].args[0] == 'Reconfigure?'
    state = {'command': builder.configure_command(), 'env': {}}
    state_file.write_text(json.dumps(state))
    mocker.patch.object(builder, 'configure_env', return_value={'X': '1'})
    confirm.reset_mock()
    builder.phase_configure()
    assert confirm.call_args_list[0].args[0] == 'Reconfigure?'


def test_clean_dry_run_keeps_tree(tmp_path: Path, mocker: MockerFixture):
    """``--clean --dry-run`` on a configured tree leaves it alone and shows the configure."""
    builder = _builder(tmp_path, mocker, clean=True)
    builder.config.dry_run = True
    cmake_dir = builder.config.cmake_dir
    cmake_dir.mkdir(parents=True)
    (cmake_dir / 'stale').write_text('')
    (cmake_dir / 'CMakeCache.txt').write_text('')
    state = json.dumps({'command': builder.configure_command(), 'env': builder.configure_env()})
    (cmake_dir / phases.STATE_FILENAME).write_text(state)
    builder.phase_configure()
    out = builder.console.export_text()
    assert '$ cmake -Wno-dev' in out
    assert 'already configured' not in out
    assert (cmake_dir / 'stale').exists()
    assert (cmake_dir / phases.STATE_FILENAME).read_text() == state


def test_disabled_groups_unset_their_keys(tmp_path: Path, mocker: MockerFixture, demo):
    assert cmake_unset(demo, 'linux', groups=['a', 'core']) == ['B', 'C']
    assert cmake_unset(demo, 'macos', groups=['b', 'core']) == []
    builder = _builder(tmp_path, mocker)
    cmd = builder.configure_command()
    unset = [a for a in cmd if a.startswith('-U')]
    assert unset == ['-UVTK_OPENGL_HAS_EGL', '-UVTK_USE_X']
    first_define = min(i for i, a in enumerate(cmd) if a.startswith(('-C', '-D')))
    assert all(cmd.index(a) < first_define for a in unset)
    builder.groups = [*builder.groups, 'headless']
    cmd = builder.configure_command()
    assert not any(a.startswith('-U') for a in cmd)


def _write_wheel(path: Path, data: bytes, year: int = 2020) -> None:
    """Write a one-file wheel whose file carries the given timestamp."""
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr(zipfile.ZipInfo('vtkmodules/lib.so', (year, 1, 1, 0, 0, 0)), data)


def test_repair_skips_a_wheel_it_already_repaired(tmp_path: Path, mocker: MockerFixture):
    """Only a wheel with different contents, or a missing result, runs the repair again."""
    builder = _builder(tmp_path, mocker, repair=True)
    cfg = builder.config
    cfg.dist_dir.mkdir(parents=True)
    cfg.wheelhouse.mkdir(parents=True)
    built = cfg.dist_dir / 'vtk-9.7.0-cp314-cp314-linux_x86_64.whl'
    repaired = cfg.wheelhouse / 'vtk-9.7.0-cp314-cp314-manylinux_2_39_x86_64.whl'
    run = mocker.patch.object(
        builder, 'run', side_effect=lambda *_, **__: _write_wheel(repaired, b'x')
    )

    def repair(data: bytes, year: int = 2020) -> None:
        _write_wheel(built, data, year)
        builder.wheel = built
        builder.phase_repair()
        assert builder.wheel == repaired

    repair(b'one')
    repair(b'one', year=2021)
    assert run.call_count == 1
    assert 'was repaired from this same wheel' in builder.console.export_text()
    repair(b'two')
    assert run.call_count == 2
    repaired.unlink()
    repair(b'two')
    assert run.call_count == 3
    (cfg.wheelhouse / phases._REPAIR_STAMP).write_text('[]')
    repair(b'two')
    assert run.call_count == 4


def test_wheel_hides_only_the_known_setuptools_warnings(tmp_path: Path, mocker: MockerFixture):
    """The filters drop two exact warnings, keep any other, and step aside for --verbose."""
    builder = _builder(tmp_path, mocker)
    builder.env['PYTHONWARNINGS'] = 'error::ResourceWarning'
    run = mocker.patch.object(builder, 'run')
    mocker.patch.object(builder, '_newest_wheel')
    builder.phase_wheel()
    filters = run.call_args.kwargs['env']['PYTHONWARNINGS']
    assert filters.split(',') == ['error::ResourceWarning', *phases._SETUPTOOLS_WARNINGS]
    builder.config.verbose = True
    builder.phase_wheel()
    assert 'PYTHONWARNINGS' not in run.call_args.kwargs['env']

    emit = (
        'import warnings\n'
        'for text, module in [\n'
        "    ('License classifiers are deprecated.', 'setuptools.dist'),\n"
        "    ('setup.py install is deprecated.', 'distutils.cmd'),\n"
        "    ('setup.py install is deprecated.', 'setuptools._distutils.cmd'),\n"
        "    ('License classifiers are deprecated.', 'mypackage'),\n"
        "    ('Another setuptools warning.', 'setuptools.dist'),\n"
        ']:\n'
        "    warnings.warn_explicit(text, UserWarning, module + '.py', 1, module)\n"
    )
    result = subprocess.run(
        [sys.executable, '-c', emit],
        env={**os.environ, 'PYTHONWARNINGS': ','.join(phases._SETUPTOOLS_WARNINGS)},
        capture_output=True,
        text=True,
        check=True,
    )
    shown = [line for line in result.stderr.splitlines() if 'UserWarning' in line]
    assert shown == [
        'mypackage.py:1: UserWarning: License classifiers are deprecated.',
        'setuptools.dist.py:1: UserWarning: Another setuptools warning.',
    ]


def test_build_wheel_repair_install_paths(tmp_path: Path, mocker: MockerFixture):
    mocker.patch.object(
        tools, 'pip_install_command', side_effect=lambda *s, args=(): ['pip', 'install', *args, *s]
    )
    builder = _builder(tmp_path, mocker, repair=True)
    mocker.patch.object(builder, 'confirm', return_value=False)
    with pytest.raises(BuildError, match='Stopped before building'):
        builder.phase_build()
    with pytest.raises(BuildError, match='Stopped before packaging'):
        builder.phase_wheel()
    builder.phase_repair()
    assert 'Skipped.' in builder.console.export_text()
    builder.phase_install()
    assert 'Not installed' in builder.console.export_text()
    with pytest.raises(BuildError, match='No wheel found'):
        builder._newest_wheel(tmp_path)

    builder = _builder(tmp_path, mocker, install=True, repair=True)
    run = mocker.patch.object(builder, 'run')
    cfg = builder.config
    cfg.dist_dir.mkdir(parents=True)
    _write_wheel(cfg.dist_dir / 'vtk-9.7.0-cp314-cp314-linux_x86_64.whl', b'')
    builder.phase_wheel()
    env = {**builder.wheel_env(), 'PYTHONWARNINGS': ','.join(phases._SETUPTOOLS_WARNINGS)}
    assert run.call_args_list[0] == mocker.call(
        [builder.python, 'setup.py', 'bdist_wheel'], cwd=cfg.cmake_dir, env=env
    )
    assert builder.wheel is not None
    assert builder.wheel.name.startswith('vtk-9.7.0')
    built = builder.wheel
    cfg.wheelhouse.mkdir(parents=True)
    _write_wheel(cfg.wheelhouse / 'vtk-9.7.0-cp314-cp314-manylinux_2_39_x86_64.whl', b'')
    builder.phase_repair()
    assert 'manylinux' in builder.wheel.name
    cmd, env = builder.repair_command(built)
    assert run.call_args_list[-1] == mocker.call(cmd, env=env)
    builder.phase_install()
    install, check = run.call_args_list[-2:]
    assert install == mocker.call(
        ['pip', 'install', '--no-deps', '--force-reinstall', str(builder.wheel)]
    )
    assert check == mocker.call([builder.python, '-c', 'import vtk'])
    builder.summary()
    out = builder.console.export_text()
    assert 'Wheel:' in out
    assert 'run the same command again' in out
    builder.config.clean = True
    builder.summary()
    assert 'the same command without --clean again' in builder.console.export_text()
    mocker.patch.object(tools, 'pip_install_command', return_value=None)
    builder.phase_install()
    assert 'install the wheel yourself' in builder.console.export_text()


def test_vswhere(tmp_path: Path, mocker: MockerFixture):
    mocker.patch.dict(tools.os.environ, {'PROGRAMFILES(X86)': '', 'PROGRAMFILES': ''})
    assert tools.vswhere_path() is None
    assert tools.find_vcvarsall() is None
    mocker.patch.dict(tools.os.environ, {'PROGRAMFILES(X86)': str(tmp_path)})
    assert tools.vswhere_path() is None
    vswhere = tmp_path / 'Microsoft Visual Studio' / 'Installer' / 'vswhere.exe'
    vswhere.parent.mkdir(parents=True)
    vswhere.write_text('')
    assert tools.vswhere_path() == vswhere
    run = mocker.patch.object(
        tools.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, stdout='')
    )
    assert tools.find_vcvarsall() is None
    install = tmp_path / 'VS'
    vcvarsall = install / 'VC' / 'Auxiliary' / 'Build' / 'vcvarsall.bat'
    vcvarsall.parent.mkdir(parents=True)
    vcvarsall.write_text('')
    run.return_value = subprocess.CompletedProcess([], 0, stdout=f'{install}\n')
    assert tools.find_vcvarsall() == vcvarsall
    assert 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64' in run.call_args.args[0]
    assert tools.find_vcvarsall('arm64') == vcvarsall
    assert 'Microsoft.VisualStudio.Component.VC.Tools.ARM64' in run.call_args.args[0]


@pytest.mark.parametrize(
    ('plat', 'arch'), [('win-arm64', 'arm64'), ('win-amd64', 'x64'), ('win32', 'x86')]
)
def test_msvc_arch_follows_the_interpreter(mocker: MockerFixture, plat, arch):
    mocker.patch.object(tools.sysconfig, 'get_platform', return_value=plat)
    assert tools.msvc_arch() == arch


@pytest.mark.usefixtures('no_tools')
def test_intro_explains_and_asks(tmp_path: Path, mocker: MockerFixture):
    console = Console(record=True, width=4000, force_terminal=False)
    Builder(_config(tmp_path, interactive=True), console).build()
    out = console.export_text()
    assert out.splitlines()[1] == 'The build runs these eight steps:'
    assert 'asks before running' not in out
    assert out.index('Proceed?') < out.index('1/8 Tools') < out.index('2/8 Source')
    mocker.patch.object(phases.sys.stdin, 'isatty', return_value=True)
    mocker.patch.object(phases.Confirm, 'ask', return_value=False)
    builder = Builder(_config(tmp_path, interactive=True, dry_run=False), console)
    with pytest.raises(BuildError, match='Stopped before the first step'):
        builder.build()
    assert 'asks before running any command; Enter takes the default' in console.export_text()


@pytest.mark.parametrize(
    ('kwargs', 'tty', 'answer', 'error'),
    [
        ({'yes': True}, False, None, None),
        ({'dry_run': True}, False, None, None),
        ({}, False, None, 'Pass --yes'),
        ({'interactive': True}, False, None, 'Pass --yes'),
        ({}, True, True, None),
        ({}, True, False, 'Stopped before the first step'),
        ({'yes': True, 'interactive': True}, True, None, None),
    ],
)
def test_confirm_start(tmp_path: Path, mocker: MockerFixture, kwargs, tty, answer, error):
    """Every run asks once before the first step unless ``--yes`` or ``--dry-run`` is given."""
    mocker.patch.object(phases.sys.stdin, 'isatty', return_value=tty)
    ask = mocker.patch.object(phases.Confirm, 'ask', return_value=answer)
    console = Console(record=True, width=4000, force_terminal=False)
    builder = Builder(_config(tmp_path, **{'dry_run': False, **kwargs}), console)
    first = mocker.patch.object(builder, 'phase_tools', side_effect=RuntimeError('started'))
    with pytest.raises(BuildError if error else RuntimeError, match=error or 'started'):
        builder.build()
    assert ask.call_count == (answer is not None)
    assert first.called == (error is None)
    assert f'go in {tmp_path / "vtk-build"}' in console.export_text()


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


def test_newest_wheel_by_mtime(tmp_path: Path):
    old, new = tmp_path / 'b-old.whl', tmp_path / 'a-new.whl'
    old.write_bytes(b'')
    new.write_bytes(b'')
    os.utime(old, (1, 1))
    builder = Builder(_config(tmp_path), Console(quiet=True))
    assert builder._newest_wheel(tmp_path) == new
    assert builder._newest_wheel(tmp_path / 'empty') is None


@pytest.mark.parametrize('os_name', OS_NAMES)
def test_wheel_env_and_repair_command(tmp_path: Path, mocker: MockerFixture, os_name):
    mocker.patch.object(phases.platform, 'machine', return_value='arm64')
    builder = _builder(tmp_path, mocker, os_name=os_name)
    cfg = builder.config
    wheel = tmp_path / 'vtk.whl'
    cmd, env = builder.repair_command(wheel)
    out = str(cfg.wheelhouse)
    expected = {
        'linux': (['auditwheel', 'repair', str(wheel), '-w', out], {}),
        'windows': (
            [
                'delvewheel',
                'repair',
                '--add-path',
                str(cfg.cmake_dir / 'bin'),
                '-w',
                out,
                str(wheel),
            ],
            {},
        ),
        'macos': (
            ['delocate-wheel', '--require-archs', 'arm64', '-w', out, str(wheel)],
            {'DYLD_LIBRARY_PATH': str(cfg.cmake_dir / 'lib')},
        ),
    }
    assert (cmd, env) == expected[os_name]
    if os_name == 'macos':
        lib = cfg.cmake_dir / 'lib' / 'vtk' / 'libvtksys-9.7.dylib'
        lib.parent.mkdir(parents=True)
        lib.write_bytes(b'')
        assert builder.repair_command(wheel)[1] == {'DYLD_LIBRARY_PATH': str(lib.parent)}
        cfg.verbose = True
        assert builder.repair_command(wheel)[0][-2:] == ['-v', str(wheel)]
    wheel_env = {'_PYTHON_HOST_PLATFORM': 'macosx-11.0-arm64', 'ARCHFLAGS': '-arch arm64'}
    assert builder.wheel_env() == (wheel_env if os_name == 'macos' else {})
    names = [t.name for t in tools.required_tools(os_name, {}, repair=True)]
    compiler = 'msvc' if os_name == 'windows' else 'c++ compiler'
    repairer = {'linux': 'auditwheel', 'macos': 'delocate-wheel', 'windows': 'delvewheel'}
    patchelf = ['patchelf'] if os_name == 'linux' else []
    assert names == ['git', 'cmake', 'ninja', compiler, repairer[os_name], *patchelf]
    assert repairer[os_name] not in [
        t.name for t in tools.required_tools(os_name, {}, repair=False)
    ]


def test_build_asks_for_jobs(tmp_path: Path, mocker: MockerFixture):
    mocker.patch.object(phases.sys.stdin, 'isatty', return_value=True)
    mocker.patch.object(phases.os, 'cpu_count', return_value=8)
    mocker.patch.object(tools, 'physical_memory', return_value=64 * 2**30)
    builder = _builder(tmp_path, mocker)
    builder.config.interactive = True
    run = mocker.patch.object(builder, 'run')
    mocker.patch.object(builder, 'hooks')
    mocker.patch.object(phases.Prompt, 'ask', side_effect=['x', '0', '4'])
    builder.phase_build()
    out = builder.console.export_text()
    assert '8 CPU cores and 64 GB of memory detected; compiling with 8 jobs.' in out
    assert run.call_args_list[0].args[0][-2:] == ['--parallel', '4']
    mocker.patch.object(phases.Prompt, 'ask', return_value='y')
    builder.phase_build()
    assert run.call_args_list[-1].args[0][-2:] == ['--parallel', '8']
    mocker.patch.object(phases.Prompt, 'ask', return_value='n')
    with pytest.raises(BuildError, match='Stopped before building'):
        builder.phase_build()


def test_msvc_environment_skips_lines_without_value(mocker: MockerFixture, tmp_path: Path):
    mocker.patch.object(
        tools.subprocess,
        'run',
        return_value=subprocess.CompletedProcess(
            '', 0, stdout='banner\nINCLUDE=C:\\inc\n'.encode('utf-16-le')
        ),
    )
    env = tools.msvc_environment(tmp_path / 'vcvarsall.bat', env={})
    assert env == {'INCLUDE': 'C:\\inc'}


def test_windows_declined_msvc(tmp_path: Path, mocker: MockerFixture):
    mocker.patch.object(tools.shutil, 'which', return_value=None)
    mocker.patch.object(tools, 'find_vcvarsall', return_value=tmp_path / 'vcvarsall.bat')
    activated = mocker.patch.object(tools, 'msvc_environment')
    builder = _builder(tmp_path, mocker, os_name='windows')
    mocker.patch.object(builder, 'confirm', return_value=False)
    builder._activate_msvc(tools.required_tools('windows', builder.manifest.tools, repair=False))
    assert not activated.called


def test_plan(tmp_path: Path):
    """The start summary says what every step does with the given settings."""
    cfg = _config(
        tmp_path,
        repair=True,
        install=True,
        jobs=3,
        defines=['X=1'],
        without=['web'],
        ref='v9.7.0',
    )
    plan = dict(Builder(cfg, Console(quiet=True)).plan())
    assert list(plan) == [
        'Tools',
        'Source',
        'Build options',
        'Configure',
        'Build',
        'Wheel',
        'Repair',
        'Install',
    ]
    assert (
        plan['Source']
        == f'clone https://gitlab.kitware.com/vtk/vtk.git at v9.7.0 into {cfg.src_dir}'
    )
    assert 'auditwheel' in plan['Tools']
    assert 'auditwheel' in plan['Repair']
    assert 'rendering' in plan['Build options']
    assert 'web' not in plan['Build options']
    assert plan['Configure'].endswith(', plus -DX=1')
    assert plan['Build'].startswith('compile with 3 parallel jobs')
    assert plan['Install'] == f'install the wheel into {sys.prefix}'
    with pytest.raises(BuildError, match='nope'):
        Builder(_config(tmp_path, with_=['nope']), Console(quiet=True)).plan()

    (cfg.src_dir / '.git').mkdir(parents=True)
    (cfg.src_dir / 'pyvista-build.toml').write_text('manifest_version = 99\n')
    plan = dict(
        Builder(
            _config(tmp_path, backend='demo', ref='main', repair=False), Console(quiet=True)
        ).plan()
    )
    assert plan['Source'] == (
        f'reuse the checkout at {cfg.src_dir}, checked out at main'
        ' (local edits stop it unless it is already there)'
    )
    assert "checkout's manifest" in plan['Build options']
    assert plan['Repair'].startswith('skipped (--no-repair)')
    assert plan['Install'].startswith('skipped')


def _path_style(path: str) -> str:
    """Return the style a narrated POSIX path gets, unlinked where it has no file URI."""
    return 'magenta' if os.name == 'nt' else f'magenta link file://{path}'


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


def test_plan_rejects_unknown_backend(tmp_path: Path, mocker: MockerFixture):
    """The start summary fails on a backend with no URL before asking to proceed."""
    builder = Builder(_config(tmp_path, backend='nosuch'), Console(quiet=True))
    with pytest.raises(BuildError, match='Unknown backend'):
        builder.plan()
    mocker.patch.object(tools, 'detect', return_value=True)
    with pytest.raises(BuildError, match='Unknown backend'):
        builder.phase_source()


def test_plan_wording_under_yes_and_source(tmp_path: Path, mocker: MockerFixture):
    """``--yes`` and ``--source`` change what the summary promises."""
    console = Console(record=True, width=4000, force_terminal=False)
    (tmp_path / 'elsewhere' / '.git').mkdir(parents=True)
    builder = Builder(_config(tmp_path, yes=True, source=tmp_path / 'elsewhere'), console)
    mocker.patch.object(builder, 'phase_tools', side_effect=RuntimeError('started'))
    with pytest.raises(RuntimeError):
        builder.build()
    assert 'Build tree and wheel go in' in console.export_text()


@pytest.mark.usefixtures('no_tools')
def test_start_covers_the_planned_installs(tmp_path: Path, mocker: MockerFixture):
    """Starting approves the installs the plan lists, and nothing else."""
    builder = Builder(_config(tmp_path, dry_run=False), Console(record=True, width=4000))
    assert builder.plan()[0][1].endswith('. Nothing to install')
    mocker.patch.object(tools, 'python_build_packages', return_value=['wheel'])
    mocker.patch.object(phases.sys.stdin, 'isatty', return_value=True)
    ask = mocker.patch.object(phases.Confirm, 'ask', return_value=True)
    assert builder.plan()[0][1].endswith(f'. Install the missing wheel into {sys.prefix}')
    builder.confirm_start()
    builder.confirm_installs(builder.install_plan(['wheel']))
    assert ask.call_count == 1
    ask.return_value = False
    with pytest.raises(BuildError, match='Nothing was installed'):
        builder.confirm_installs(builder.install_plan(['wheel', 'ninja']))
    builder.config.interactive = True
    with pytest.raises(BuildError, match='Nothing was installed'):
        builder.confirm_installs(builder.install_plan(['wheel']))


def test_wheel_env_takes_the_define(tmp_path: Path, mocker: MockerFixture):
    """A ``-D CMAKE_OSX_DEPLOYMENT_TARGET`` sets the wheel's platform tag."""
    builder = _builder(
        tmp_path, mocker, os_name='macos', defines=['CMAKE_OSX_DEPLOYMENT_TARGET:STRING=12.0']
    )
    assert builder.wheel_env()['_PYTHON_HOST_PLATFORM'].startswith('macosx-12.0-')


def test_narration_escapes_paths(tmp_path: Path):
    """Brackets in a path are printed, not read as markup."""
    console = Console(record=True, width=4000, force_terminal=False)
    cfg = _config(tmp_path, build_dir=tmp_path / 'b[x]')
    Builder(cfg, console).summary()
    assert 'b[x]' in console.export_text()


def test_default_jobs_follow_memory(tmp_path: Path, mocker: MockerFixture):
    """The build step and the plan report a job count limited by memory."""
    gib = 2**30
    mocker.patch.object(phases.os, 'cpu_count', return_value=16)
    mocker.patch.object(tools, 'physical_memory', return_value=8 * gib)
    builder = _builder(tmp_path, mocker)
    mocker.patch.object(builder, 'run')
    mocker.patch.object(builder, 'hooks')
    builder.config.yes = True
    builder.phase_build()
    out = builder.console.export_text()
    assert '16 CPU cores and 8 GB of memory detected; compiling with 4 jobs, one per 2 GB' in out
    assert dict(builder.plan())['Build'].startswith(
        'compile with 4 parallel jobs (one per 2 GB of memory)'
    )
    builder.config.jobs = 6
    assert dict(builder.plan())['Build'].startswith('compile with 6 parallel jobs, logging')


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


def test_prompting_checks_the_selection_first(tmp_path: Path, mocker: MockerFixture):
    """An unknown ``--with`` group stops an interactive build before the first question."""
    mocker.patch.object(phases.sys.stdin, 'isatty', return_value=True)
    ask = mocker.patch.object(phases.Confirm, 'ask', return_value=True)
    cfg = _config(tmp_path, interactive=True, dry_run=False, with_=['nope'])
    builder = Builder(cfg, Console(quiet=True))
    tools_phase = mocker.spy(builder, 'phase_tools')
    with pytest.raises(BuildError, match='Unknown group'):
        builder.build()
    assert not ask.called
    assert not tools_phase.called


def test_failed_post_configure_hook_reruns_configure(tmp_path: Path, mocker: MockerFixture):
    """The configure state is saved only after the hooks pass, so a failed hook runs again."""
    builder = _builder(tmp_path, mocker)
    cmake_dir = builder.config.cmake_dir
    run = mocker.patch.object(builder, 'run')
    hooks = mocker.patch.object(builder, 'hooks', side_effect=BuildError('hook failed'))
    with pytest.raises(BuildError, match='hook failed'):
        builder.phase_configure()
    assert not (cmake_dir / phases.STATE_FILENAME).exists()
    cmake_dir.mkdir(parents=True)
    (cmake_dir / 'CMakeCache.txt').write_text('')
    hooks.side_effect = None
    builder.phase_configure()
    assert run.call_count == 2
    assert hooks.call_count == 2
    assert (cmake_dir / phases.STATE_FILENAME).is_file()


def test_build_runs_in_the_configure_env(tmp_path: Path, mocker: MockerFixture):
    """``cmake --build`` gets the configure environment, so a self-reconfigure keeps it."""
    builder = _builder(tmp_path, mocker, os_name='macos', yes=True)
    run = mocker.patch.object(builder, 'run')
    mocker.patch.object(builder, 'hooks')
    builder.phase_build()
    env = run.call_args.kwargs['env']
    assert env == builder.configure_env()
    assert '_PYTHON_HOST_PLATFORM' in env


def test_macos_deployment_target_fallback(tmp_path: Path, mocker: MockerFixture):
    """Without a deployment target from the manifest or ``-D``, cmake gets the tag's one."""
    define = f'-DCMAKE_OSX_DEPLOYMENT_TARGET={phases.DEFAULT_MACOS_TARGET}'
    builder = _builder(tmp_path, mocker, os_name='macos')
    assert builder.configure_command().count(define) == 1
    manifest = builder.manifest
    builder._manifest = replace(manifest, cmake_os={**manifest.cmake_os, 'macos': ()})
    assert define in builder.configure_command()
    platform_tag = builder.wheel_env()['_PYTHON_HOST_PLATFORM']
    assert platform_tag.startswith(f'macosx-{phases.DEFAULT_MACOS_TARGET}-')
    builder.config.defines = ['CMAKE_OSX_DEPLOYMENT_TARGET=12.0']
    cmd = builder.configure_command()
    assert define not in cmd
    assert '-DCMAKE_OSX_DEPLOYMENT_TARGET=12.0' in cmd
    linux = _builder(tmp_path, mocker)
    assert not any('CMAKE_OSX_DEPLOYMENT_TARGET' in arg for arg in linux.configure_command())


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
