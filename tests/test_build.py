"""Tests for the backend build engine and the manifest machinery behind it."""

from __future__ import annotations

import re
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest
from rich.console import Console

from pyvista._build import BuildConfig
from pyvista._build import Builder
from pyvista._build import BuildError
from pyvista._build import ManifestError
from pyvista._build import load_manifest
from pyvista._build import manylinux
from pyvista._build import phases
from pyvista._build import tools
from pyvista._build.manifest import Option
from pyvista._build.manifest import cmake_entries
from pyvista._build.manifest import parse_manifest
from pyvista._build.manifest import resolve_groups
from pyvista._build.manifest import resolve_options
from pyvista._build.registry import find_manifest
from pyvista._build.registry import known_backends
from pyvista._build.registry import shipped_manifest

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

OS_NAMES = ['linux', 'macos', 'windows']

MINIMAL_MANIFEST = {
    'manifest_version': 1,
    'backend': {'name': 'demo', 'import_name': 'demo', 'default_ref': 'v1.0'},
    'groups': {
        'a': {'help': 'A', 'default': True, 'cmake': ['A=ON'], 'cmake_off': ['A=OFF']},
        'b': {'help': 'B', 'default': False, 'cmake': ['B=ON'], 'conflicts': ['c']},
        'c': {
            'help': 'C',
            'default': False,
            'cmake': ['C=ON'],
            'conflicts': ['b'],
            'os': ['linux'],
        },
        'core': {'help': 'Core', 'required': True, 'cmake': ['CORE=ON']},
    },
    'options': {
        'lto': {'help': 'LTO', 'default': False, 'cmake': 'LTO={value}'},
        'smp': {
            'help': 'SMP',
            'default': 'STDThread',
            'cmake': 'SMP={value}',
            'choices': ['STDThread', 'TBB'],
        },
        'jobs': {'help': 'Jobs', 'default': 2, 'cmake': 'JOBS={value}'},
    },
    'hooks': {'post_build': ['{python} prune.py {build_dir}']},
}


@pytest.fixture
def demo():
    return parse_manifest(MINIMAL_MANIFEST)


def test_shipped_vtk_manifest_loads():
    path = shipped_manifest('vtk')
    assert path is not None
    manifest = load_manifest(path)
    assert manifest.name == 'vtk'
    assert manifest.import_name == 'vtkmodules'
    assert 'rendering' in manifest.groups
    assert manifest.groups['headless-egl'].os == ('linux',)
    assert find_manifest('vtk', None).path == path


@pytest.mark.parametrize('os_name', OS_NAMES)
def test_vtk_manifest_defaults_match_ci_flags(os_name):
    manifest = find_manifest('vtk', None)
    groups = resolve_groups(manifest, os_name)
    entries = cmake_entries(
        manifest, os_name, groups=groups, options=resolve_options(manifest, None)
    )
    ci_flags = {
        'VTK_WHEEL_BUILD=ON',
        'VTK_WRAP_PYTHON=ON',
        'VTK_GROUP_ENABLE_Rendering=WANT',
        'VTK_BUILD_TESTING=OFF',
        'VTK_BUILD_DOCUMENTATION=OFF',
        'VTK_BUILD_EXAMPLES=OFF',
        'VTK_USE_PCH=OFF',
        'VTK_DISPATCH_SOA_ARRAYS=ON',
        'VTK_WRAP_SERIALIZATION=ON',
        'VTK_BUILD_PYI_FILES=ON',
        'VTK_JPEG_ENABLE_SIMD=ON',
        'VTK_MODULE_ENABLE_VTK_FiltersParallelDIY2=YES',
        'VTK_MODULE_ENABLE_VTK_RenderingMatplotlib=YES',
        'VTK_MODULE_ENABLE_VTK_IOImport=YES',
        'VTK_MODULE_ENABLE_VTK_IOParallelExodus=YES',
        'VTK_MODULE_ENABLE_VTK_IOXdmf2=YES',
        'VTK_MODULE_ENABLE_VTK_WebCore=YES',
        'VTK_MODULE_ENABLE_VTK_WebGLExporter=YES',
        'VTK_MODULE_ENABLE_VTK_WebPython=YES',
        'VTK_SMP_IMPLEMENTATION_TYPE=STDThread',
    }
    if os_name == 'macos':
        ci_flags.add('CMAKE_OSX_DEPLOYMENT_TARGET=11.0')
    assert set(entries) == ci_flags
    assert not any('EGL' in e or 'OSMESA' in e for e in entries)


def test_resolve_groups_defaults(demo):
    assert resolve_groups(demo, 'linux') == ['a', 'core']
    assert resolve_groups(demo, 'linux', minimal=True) == ['core']
    assert resolve_groups(demo, 'macos', all_=True) == ['a', 'b', 'core']
    assert resolve_groups(demo, 'linux', with_=['b'], without=['a']) == ['b', 'core']


def test_resolve_groups_errors(demo):
    with pytest.raises(ManifestError, match='conflicts'):
        resolve_groups(demo, 'linux', all_=True)
    with pytest.raises(ManifestError, match='required'):
        resolve_groups(demo, 'linux', without=['core'])
    with pytest.raises(ManifestError, match='Unknown group'):
        resolve_groups(demo, 'linux', with_=['nope'])
    with pytest.raises(ManifestError, match='not available on macos'):
        resolve_groups(demo, 'macos', with_=['c'])
    with pytest.raises(ManifestError, match='mutually exclusive'):
        resolve_groups(demo, 'linux', all_=True, minimal=True)


def test_cmake_entries_use_off_list(demo):
    assert cmake_entries(demo, 'linux', groups=['core'], options={}) == ['A=OFF', 'CORE=ON']
    assert cmake_entries(demo, 'linux', groups=['a', 'core'], options={}) == ['A=ON', 'CORE=ON']


def test_resolve_options(demo):
    values = resolve_options(demo, ['lto=on', 'smp=TBB', 'jobs=8'])
    assert values == {'lto': True, 'smp': 'TBB', 'jobs': 8}
    entries = cmake_entries(demo, 'linux', groups=['core'], options=values)
    assert entries[-3:] == ['LTO=ON', 'SMP=TBB', 'JOBS=8']
    with pytest.raises(ManifestError, match='boolean'):
        resolve_options(demo, ['lto=maybe'])
    with pytest.raises(ManifestError, match='one of'):
        resolve_options(demo, ['smp=OpenMP'])
    with pytest.raises(ManifestError, match='integer'):
        resolve_options(demo, ['jobs=many'])
    with pytest.raises(ManifestError, match='Unknown option'):
        resolve_options(demo, ['nope=1'])


@pytest.mark.parametrize(
    ('patch', 'match'),
    [
        ({'manifest_version': 2}, 'manifest_version'),
        ({'backend': {'name': 'x'}}, 'import_name'),
        ({'groups': {'g': {'default': True}}}, 'help'),
        ({'groups': {'g': {'help': 'x', 'os': ['beos']}}}, 'unknown os'),
        ({'groups': {'g': {'help': 'x', 'conflicts': ['zz']}}}, 'unknown group'),
        ({'options': {'o': {'help': 'x', 'default': True, 'cmake': 'X=ON'}}}, 'placeholder'),
        ({'options': {'o': {'help': 'x', 'default': [], 'cmake': 'X={value}'}}}, 'default'),
        ({'hooks': {'pre_build': []}}, 'unknown hook'),
        ({'cmake': {'init_cache': {'beos': 'x'}}}, 'init_cache'),
    ],
)
def test_parse_manifest_errors(patch, match):
    data = dict(MINIMAL_MANIFEST, **patch)
    with pytest.raises(ManifestError, match=match):
        parse_manifest(data)


def test_load_manifest_bad_toml(tmp_path: Path):
    path = tmp_path / 'pyvista-build.toml'
    path.write_text('manifest_version = [')
    with pytest.raises(ManifestError, match=re.escape(str(path))):
        load_manifest(path)


def test_find_manifest_prefers_tree(tmp_path: Path):
    (tmp_path / 'pyvista-build.toml').write_text(
        'manifest_version = 1\n[backend]\nname = "vtk"\nimport_name = "x"\ndefault_ref = "r"\n'
    )
    assert find_manifest('vtk', tmp_path).import_name == 'x'
    with pytest.raises(ManifestError, match='ships no manifest'):
        find_manifest('nope', tmp_path.parent / 'missing')


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


@pytest.mark.usefixtures('no_tools')
@pytest.mark.parametrize('os_name', OS_NAMES)
def test_dry_run_commands(tmp_path: Path, os_name):
    console = Console(record=True, width=4000, force_terminal=False)
    builder = Builder(_config(tmp_path, os_name=os_name, defines=['FOO=1'], ref='v9.7.0'), console)
    assert builder.build() is None
    out = console.export_text()
    assert out.startswith('Building backend: vtk')
    assert 'git clone --filter=blob:none https://gitlab.kitware.com/vtk/vtk.git' in out
    assert 'checkout v9.7.0' in out
    assert '-DFOO=1' in out
    assert '-DVTK_VERSION_SUFFIX= ' in out
    repairer = {
        'linux': 'auditwheel repair',
        'macos': 'delocate-wheel',
        'windows': 'delvewheel repair',
    }
    assert repairer[os_name] in out
    assert 'setup.py bdist_wheel' in out
    assert '--force-reinstall' in out  # printed as the install hint
    launcher = 'sccache' if os_name == 'windows' else 'ccache'
    assert f'-DCMAKE_CXX_COMPILER_LAUNCHER={launcher}' in out


@pytest.mark.usefixtures('no_tools')
def test_dry_run_flags(tmp_path: Path):
    console = Console(record=True, width=4000, force_terminal=False)
    cfg = _config(
        tmp_path,
        minimal=True,
        compiler_cache='none',
        repair=False,
        build_type='Debug',
        jobs=3,
        ref='main',
        version_suffix='',
        options=['smp=TBB'],
    )
    Builder(cfg, console).build()
    out = console.export_text()
    assert 'Compiler cache: off' in out
    assert 'COMPILER_LAUNCHER' not in out
    assert '-DCMAKE_BUILD_TYPE=Debug' in out
    assert '--parallel 3' in out
    assert '-DVTK_GROUP_ENABLE_Rendering=NO' in out
    assert '-DVTK_SMP_IMPLEMENTATION_TYPE=TBB' in out
    assert '-DVTK_VERSION_SUFFIX= ' in out
    assert 'Skipped (--no-repair)' in out


@pytest.mark.usefixtures('no_tools')
def test_dry_run_dev_suffix(tmp_path: Path):
    console = Console(record=True, width=4000, force_terminal=False)
    Builder(_config(tmp_path, ref='main'), console).build()
    assert '-DVTK_VERSION_SUFFIX=dev0+g<sha>' in console.export_text()


def test_unknown_backend(tmp_path: Path):
    with pytest.raises(BuildError, match='Unknown backend'):
        Builder(_config(tmp_path, backend='nope'), Console(quiet=True)).build()


def test_missing_git(tmp_path: Path, mocker: MockerFixture):
    mocker.patch.object(tools.shutil, 'which', return_value=None)
    with pytest.raises(BuildError, match='git was not found'):
        Builder(_config(tmp_path), Console(quiet=True)).build()


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
    Builder(_config(tmp_path), console).build()
    out = console.export_text()
    assert 'Missing Python packages: cmake>=3.22, auditwheel, wheel' in out
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
    assert 'too old' in out
    assert f'$ {phases._join(["pip", "install", "cmake>=3.22"])}' in out


@pytest.mark.usefixtures('no_tools')
def test_explicit_cache_missing(tmp_path: Path, mocker: MockerFixture):
    mocker.patch.object(tools, 'compiler_cache_tool', return_value=None)
    with pytest.raises(BuildError, match='sccache was not found'):
        Builder(_config(tmp_path, compiler_cache='sccache'), Console(quiet=True)).build()


@pytest.mark.usefixtures('no_tools')
def test_hooks_and_init_cache(tmp_path: Path):
    src = tmp_path / 'src'
    (src / '.git').mkdir(parents=True)
    (src / 'pyvista-build.toml').write_text(
        'manifest_version = 1\n'
        '[backend]\nname = "demo"\nimport_name = "demo"\ndefault_ref = "v1"\n'
        '[cmake]\ninit_cache = { linux = "ci/linux.cmake" }\n'
        '[hooks]\npost_build = ["{python} prune.py {build_dir}"]\n'
    )
    console = Console(record=True, width=4000, force_terminal=False)
    Builder(_config(tmp_path, backend='demo', source=src), console).build()
    out = console.export_text()
    assert f'-C {src / "ci" / "linux.cmake"}' in out
    assert f'prune.py {tmp_path / "vtk-build" / "build"}' in out
    assert 'git clone' not in out


def test_existing_checkout_dirty(tmp_path: Path, mocker: MockerFixture):
    src = tmp_path / 'vtk-build' / 'src'
    (src / '.git').mkdir(parents=True)
    mocker.patch.object(tools.shutil, 'which', side_effect=lambda name, **_: f'/usr/bin/{name}')
    mocker.patch.object(
        subprocess,
        'run',
        return_value=subprocess.CompletedProcess([], 0, stdout=' M file.cxx\n', stderr=''),
    )
    with pytest.raises(BuildError, match='local changes'):
        Builder(_config(tmp_path, ref='v9.7.0'), Console(quiet=True)).build()


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
    Builder(_config(tmp_path), console).build()
    assert 'differ from the last configure' in console.export_text()


def test_pip_install_command(mocker: MockerFixture):
    mocker.patch.object(tools.importlib.util, 'find_spec', return_value=object())
    cmd = tools.pip_install_command('cmake', args=('--no-deps',))
    assert cmd[1:] == ['-m', 'pip', 'install', '--no-deps', 'cmake']
    mocker.patch.object(tools.importlib.util, 'find_spec', return_value=None)
    mocker.patch.object(tools.shutil, 'which', return_value=None)
    assert tools.pip_install_command('cmake') is None


@pytest.mark.parametrize(
    ('choice', 'os_name', 'expected'),
    [
        ('auto', 'linux', 'ccache'),
        ('auto', 'windows', 'sccache'),
        ('sccache', 'macos', 'sccache'),
        ('none', 'linux', None),
    ],
)
def test_compiler_cache_tool(mocker: MockerFixture, choice, os_name, expected):
    mocker.patch.object(tools.shutil, 'which', side_effect=lambda name, **_: f'/usr/bin/{name}')
    tool = tools.compiler_cache_tool(choice, os_name)
    assert (tool.name if tool else None) == expected


def test_env_options_and_version_command(tmp_path: Path, mocker: MockerFixture):
    src = tmp_path / 'src'
    (src / '.git').mkdir(parents=True)
    (src / 'pyvista-build.toml').write_text(
        'manifest_version = 1\n'
        '[backend]\nname = "demo"\nimport_name = "demo"\ndefault_ref = "v1"\n'
        'version_command = "{python} version.py"\n'
        '[options.lto]\nhelp = "LTO"\ndefault = false\nenv = "DEMO_LTO"\n'
        '[options.abi3]\nhelp = "abi3"\ndefault = true\ncmake = "DEMO_ABI3={value}"\n'
    )
    mocker.patch.object(tools.shutil, 'which', side_effect=lambda name, **_: f'/usr/bin/{name}')
    mocker.patch.object(tools, 'executable_version', return_value='99.0')
    mocker.patch.object(tools, 'python_build_packages', return_value=[])
    console = Console(record=True, width=4000, force_terminal=False)
    Builder(_config(tmp_path, backend='demo', source=src, options=['lto=1']), console).build()
    out = console.export_text()
    assert '$ DEMO_LTO=1 cmake' in out
    assert '-DDEMO_ABI3=ON' in out
    assert '-DDEMO_LTO' not in out
    assert '-DVTK_VERSION_SUFFIX=<version_command>' in out


def test_option_needs_cmake_or_env():
    base = {'help': 'x', 'default': True}
    with pytest.raises(ManifestError, match='exactly one'):
        parse_manifest(dict(MINIMAL_MANIFEST, options={'o': base}))
    with pytest.raises(ManifestError, match='exactly one'):
        parse_manifest(
            dict(MINIMAL_MANIFEST, options={'o': dict(base, cmake='X={value}', env='X')})
        )


def test_known_backends():
    assert known_backends() == ['cvista', 'vtk']


@pytest.mark.usefixtures('no_tools')
def test_dry_run_needs_checkout_without_shipped_manifest(tmp_path: Path):
    cfg = _config(tmp_path, backend='demo', url='https://example.invalid/demo.git')
    with pytest.raises(BuildError, match='--dry-run needs an existing checkout'):
        Builder(cfg, Console(quiet=True)).build()


@pytest.mark.usefixtures('no_tools')
def test_cloned_tree_without_manifest(tmp_path: Path, mocker: MockerFixture):
    cfg = _config(tmp_path, backend='demo', url='https://example.invalid/demo.git', dry_run=False)
    builder = Builder(cfg, Console(quiet=True))
    mocker.patch.object(
        builder, 'git', side_effect=lambda *_a, **_k: cfg.src_dir.mkdir(parents=True)
    )
    with pytest.raises(BuildError, match=r'No pyvista-build\.toml'):
        builder.phase_source()


def test_version_command_failure(tmp_path: Path, mocker: MockerFixture):
    src = tmp_path / 'src'
    (src / '.git').mkdir(parents=True)
    (src / 'pyvista-build.toml').write_text(
        'manifest_version = 1\n'
        '[backend]\nname = "demo"\nimport_name = "demo"\ndefault_ref = "v1"\n'
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


def test_msvc_environment(mocker: MockerFixture, tmp_path: Path):
    run = mocker.patch.object(
        tools.subprocess,
        'run',
        return_value=subprocess.CompletedProcess(
            '', 0, stdout='INCLUDE=C:\\inc\nPATH=C:\\vc;old\n'
        ),
    )
    vcvarsall = tmp_path / 'vcvarsall.bat'
    before = {'PATH': 'old', 'KEEP': '1'}
    env = tools.msvc_environment(vcvarsall, env=before)
    assert env == {'PATH': 'C:\\vc;old', 'KEEP': '1', 'INCLUDE': 'C:\\inc'}
    assert before == {'PATH': 'old', 'KEEP': '1'}
    assert run.call_args.args[0] == f'"{vcvarsall}" x64 >nul && set'
    assert run.call_args.kwargs['shell'] is True


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
    cwd = tmp_path / 'made'
    builder.run(
        [sys.executable, '-c', 'import os; print(os.environ["DEMO"])'], cwd=cwd, env={'DEMO': 'x'}
    )
    assert cwd.is_dir()
    builder.run([sys.executable, '-c', 'raise SystemExit(3)'], fatal=False)
    with pytest.raises(BuildError, match='exit status 3'):
        builder.run([sys.executable, '-c', 'raise SystemExit(3)'])
    assert '--yes' in builder.console.export_text() or builder.confirm('q?')


def test_run_logged_filters_noise(tmp_path: Path, mocker: MockerFixture):
    builder = _builder(tmp_path, mocker)
    log = tmp_path / 'build.log'
    script = tmp_path / 'noisy.py'
    script.write_text(
        'print("ok line"); print("x.cxx(1): warning C4244: conversion"); raise SystemExit(1)'
    )
    with pytest.raises(BuildError):
        builder.run([sys.executable, str(script)], log=log)
    out = builder.console.export_text()
    assert 'ok line' in out
    assert 'C4244' not in out
    assert 'C4244' in log.read_text()
    assert 'full log is in' in out


def test_source_not_a_checkout(tmp_path: Path, mocker: MockerFixture):
    builder = _builder(tmp_path, mocker, source=tmp_path / 'plain')
    (tmp_path / 'plain').mkdir()
    mocker.patch.object(tools, 'detect', return_value=True)
    with pytest.raises(BuildError, match='not a git checkout'):
        builder.phase_source()


def test_existing_checkout_update_and_ref(tmp_path: Path, mocker: MockerFixture):
    src = tmp_path / 'src'
    (src / '.git').mkdir(parents=True)
    builder = _builder(tmp_path, mocker, source=src, update=True)
    git = mocker.patch.object(builder, 'git', return_value='')
    builder._existing_checkout(src)
    calls = [c.args for c in git.call_args_list]
    assert ('fetch', '--tags', 'origin') in calls
    assert any('checkout' in c and 'v9.7.0' in c for c in calls)
    builder = _builder(tmp_path, mocker, source=src, ref='main')
    git = mocker.patch.object(builder, 'git', return_value='')
    builder._existing_checkout(src)
    assert any('checkout' in c and 'main' in c for c in [c.args for c in git.call_args_list])


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
    mocker.patch.object(builder, 'confirm', return_value=False)
    with pytest.raises(BuildError, match='Cannot build without them'):
        builder.phase_tools()
    mocker.patch.object(builder, 'confirm', return_value=True)
    mocker.patch.object(builder, 'run')
    with pytest.raises(BuildError, match='Still not usable after install: ninja'):
        builder.phase_tools()


def test_windows_msvc_and_nasm(tmp_path: Path, mocker: MockerFixture):
    present = {'git', 'cmake', 'ninja', 'delvewheel'}

    def which(name, path=None, **_):
        if name == 'cl':
            return 'C:/vs/cl' if path and 'C:/vs' in path else None
        return f'C:/t/{name}' if name in present else None

    mocker.patch.object(tools.shutil, 'which', side_effect=which)
    mocker.patch.object(tools, 'executable_version', return_value='99.0')
    mocker.patch.object(tools, 'python_build_packages', return_value=[])
    mocker.patch.object(tools, 'find_vcvarsall', return_value=None)
    builder = _builder(tmp_path, mocker, os_name='windows', compiler_cache='none')
    with pytest.raises(BuildError, match='msvc'):
        builder.phase_tools()
    vcvarsall = tmp_path / 'vcvarsall.bat'
    mocker.patch.object(tools, 'find_vcvarsall', return_value=vcvarsall)
    activated = mocker.patch.object(
        tools, 'msvc_environment', return_value={'PATH': 'C:/vs;C:/t', 'INCLUDE': 'x'}
    )
    builder = _builder(tmp_path, mocker, os_name='windows', compiler_cache='none')
    builder.phase_tools()
    activated.assert_called_once()
    assert builder.env['INCLUDE'] == 'x'
    assert 'nasm not found; assembler-optimised' in builder.console.export_text()
    script = builder.config.src_dir / '.gitlab' / 'ci' / 'download_nasm.cmake'
    script.parent.mkdir(parents=True)
    script.write_text('')
    run = mocker.patch.object(builder, 'run')
    builder._activate_nasm()
    assert run.call_args.args[0] == ['cmake', '-P', str(script)]
    assert builder.env['PATH'].startswith(str(builder.config.src_dir / '.gitlab' / 'nasm'))


def test_compiler_cache_auto_missing(tmp_path: Path, mocker: MockerFixture):
    builder = _builder(tmp_path, mocker)
    mocker.patch.object(tools, 'compiler_cache_tool', return_value=None)
    builder._activate_compiler_cache()
    assert 'No ccache/sccache on PATH' in builder.console.export_text()


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
    with pytest.raises(BuildError, match='Keeping the old configuration'):
        builder.phase_configure()
    builder = _builder(tmp_path, mocker, reconfigure=True)
    mocker.patch.object(
        builder, 'confirm', side_effect=lambda q, **_: not q.startswith('Run cmake')
    )
    with pytest.raises(BuildError, match='Stopped before configure'):
        builder.phase_configure()


def test_build_wheel_repair_install_paths(tmp_path: Path, mocker: MockerFixture):
    builder = _builder(tmp_path, mocker)
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

    builder = _builder(tmp_path, mocker, install=True)
    run = mocker.patch.object(builder, 'run')
    cfg = builder.config
    cfg.dist_dir.mkdir(parents=True)
    (cfg.dist_dir / 'vtk-9.7.0-cp314-cp314-linux_x86_64.whl').write_bytes(b'')
    builder.phase_wheel()
    assert builder.wheel is not None
    assert builder.wheel.name.startswith('vtk-9.7.0')
    cfg.wheelhouse.mkdir(parents=True)
    (cfg.wheelhouse / 'vtk-9.7.0-cp314-cp314-manylinux_2_39_x86_64.whl').write_bytes(b'')
    builder.phase_repair()
    assert 'manylinux' in builder.wheel.name
    assert 'auditwheel' in run.call_args_list[-1].args[0]
    builder.phase_install()
    assert '--force-reinstall' in run.call_args.args[0]
    builder.summary()
    assert 'Wheel:' in builder.console.export_text()
    mocker.patch.object(tools, 'pip_install_command', return_value=None)
    builder.phase_install()
    assert 'install the wheel yourself' in builder.console.export_text()


def test_tool_helpers(mocker: MockerFixture):
    mocker.patch.object(tools.platform, 'system', return_value='Darwin')
    assert tools.host_os() == 'macos'
    mocker.patch.object(tools.platform, 'system', return_value='Windows')
    assert tools.host_os() == 'windows'
    mocker.patch.object(tools.platform, 'system', return_value='Linux')
    assert tools.host_os() == 'linux'
    assert tools.executable_version('/definitely/not/here') is None
    assert tools.executable_version(sys.executable) is not None
    mocker.patch.object(
        tools.subprocess,
        'run',
        return_value=subprocess.CompletedProcess([], 0, stdout='no digits'),
    )
    assert tools.executable_version('x') is None
    for os_name in OS_NAMES:
        assert tools.compiler_cache_instructions(os_name)
        assert tools.compiler_instructions(os_name)
    tool = tools.Tool('t', ('t',), spec='>=1')
    mocker.patch.object(tools.shutil, 'which', return_value='/usr/bin/t')
    mocker.patch.object(tools, 'executable_version', return_value=None)
    assert tools.detect(tool)
    mocker.patch.object(tools, 'executable_version', return_value='not.a.version')
    assert tools.detect(tool)
    assert isinstance(tools.python_build_packages(), list)


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


def test_option_coerce_and_schema_errors():
    flag = Option('f', help='', default=True, cmake='F={value}')
    assert flag.coerce('on') is True
    assert flag.coerce('off') is False
    with pytest.raises(ManifestError, match='expects a boolean'):
        flag.coerce('maybe')
    count = Option('n', help='', default=2, cmake='N={value}')
    with pytest.raises(ManifestError, match='expects an integer'):
        count.coerce('two')
    with pytest.raises(ManifestError, match="'name' must be str"):
        parse_manifest(
            dict(MINIMAL_MANIFEST, backend={'name': 1, 'import_name': 'd', 'default_ref': 'v'})
        )
    with pytest.raises(ManifestError, match='must be a list of strings'):
        parse_manifest(dict(MINIMAL_MANIFEST, groups={'g': {'help': 'x', 'cmake': 'A=ON'}}))
    with pytest.raises(ManifestError, match=r'\[tools\]'):
        parse_manifest(dict(MINIMAL_MANIFEST, tools={'cmake': 3}))
    with pytest.raises(ManifestError, match=r'\[cmake\] must be a table'):
        parse_manifest(dict(MINIMAL_MANIFEST, cmake=[]))


@pytest.mark.usefixtures('no_tools')
def test_intro_explains_and_asks(tmp_path: Path, mocker: MockerFixture):
    console = Console(record=True, width=4000, force_terminal=False)
    Builder(_config(tmp_path, interactive=True), console).build()
    out = console.export_text()
    assert 'eight steps' in out
    assert out.index('Proceed?') < out.index('1/8 Source')
    builder = Builder(_config(tmp_path, dry_run=False), Console(quiet=True))
    mocker.patch.object(builder, 'confirm', return_value=False)
    with pytest.raises(BuildError, match='Stopped before the first step'):
        builder.build()


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
        '[backend]\nname = "demo"\nimport_name = "demo"\ndefault_ref = "v1"\n'
        'version_command = "{python} version.py"\n'
    )
    builder = Builder(
        _config(tmp_path, backend='demo', source=src, dry_run=False), Console(quiet=True)
    )
    builder._manifest = find_manifest('demo', src)
    mocker.patch.object(
        subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, stdout='5\n', stderr='')
    )
    assert builder.version_suffix() == '5'


def test_compiler_cache_tool_missing(mocker: MockerFixture):
    mocker.patch.object(tools.shutil, 'which', return_value=None)
    assert tools.compiler_cache_tool('sccache', 'linux') is None
    assert tools.compiler_cache_tool('auto', 'linux') is None


def test_build_asks_for_jobs(tmp_path: Path, mocker: MockerFixture):
    mocker.patch.object(phases.sys.stdin, 'isatty', return_value=True)
    mocker.patch.object(phases.os, 'cpu_count', return_value=8)
    builder = _builder(tmp_path, mocker)
    builder.config.interactive = True
    run = mocker.patch.object(builder, 'run')
    mocker.patch.object(builder, 'hooks')
    mocker.patch.object(phases.Prompt, 'ask', side_effect=['x', '0', '4'])
    builder.phase_build()
    assert '8 CPU cores detected; compiling with 8 jobs.' in builder.console.export_text()
    assert run.call_args_list[0].args[0][-2:] == ['--parallel', '4']
    mocker.patch.object(phases.Prompt, 'ask', return_value='y')
    builder.phase_build()
    assert run.call_args_list[-1].args[0][-2:] == ['--parallel', '8']
    mocker.patch.object(phases.Prompt, 'ask', return_value='n')
    with pytest.raises(BuildError, match='Stopped before building'):
        builder.phase_build()


def test_manifest_manylinux_packages(tmp_path: Path):
    path = tmp_path / 'pyvista-build.toml'
    path.write_text(
        'manifest_version = 1\n'
        '[backend]\nname = "demo"\nimport_name = "demo"\ndefault_ref = "v1"\n'
        '[manylinux]\npackages = ["libX11-devel"]\n'
    )
    assert load_manifest(path).manylinux_packages == ('libX11-devel',)
    path.write_text(
        'manifest_version = 1\nmanylinux = 1\n'
        '[backend]\nname = "demo"\nimport_name = "demo"\ndefault_ref = "v1"\n'
    )
    with pytest.raises(ManifestError, match=r'\[manylinux\] must be a table'):
        load_manifest(path)
    assert find_manifest('vtk', None).manylinux_packages == (
        'libX11-devel',
        'libXcursor-devel',
        'mesa-libGL-devel',
    )


def test_manylinux_helpers(tmp_path: Path, mocker: MockerFixture):
    mocker.patch.object(manylinux.platform, 'machine', return_value='arm64')
    assert manylinux.default_image().endswith('_aarch64')
    mocker.patch.object(manylinux.platform, 'machine', return_value='AMD64')
    assert manylinux.default_image().endswith('_x86_64')
    tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
    assert manylinux.container_python() == f'/opt/python/{tag}-{tag}/bin/python'
    assert manylinux.pyvista_requirement(tmp_path) == manylinux.PYVISTA
    mocker.patch.object(manylinux.pv, '__version__', '0.50.0')
    assert manylinux.pyvista_requirement(None) == 'pyvista==0.50.0'
    mocker.patch.object(manylinux.pv, '__version__', '0.51.dev0')
    assert manylinux.pyvista_requirement(None) == 'pyvista'
    script = manylinux.container_script(
        python='/opt/py',
        packages=['a-devel', 'b'],
        pyvista='/pyvista',
        build_args=['vtk', '--build-dir', '/work'],
        ccache=True,
        owner='1000:1000',
        paths=['/work', '/cache'],
    )
    assert 'dnf install -y --setopt=install_weak_deps=False a-devel b' in script
    assert "git config --global --add safe.directory '*'" in script
    assert 'yum install -y a-devel b' in script
    assert 'ccache-4.14-linux-x86_64-musl-static' in script
    assert manylinux.CCACHE_SHA256 in script
    assert '"$PY" -m pip install --quiet --only-binary :all: /pyvista' in script
    assert '"$PY" -m pyvista build vtk --build-dir /work' in script
    assert script.endswith('chown -R 1000:1000 /work /cache')
    script = manylinux.container_script(
        python='/opt/py',
        packages=[],
        pyvista='pyvista',
        build_args=['vtk'],
        ccache=False,
        owner=None,
        paths=['/work'],
    )
    assert 'dnf' not in script
    assert 'ccache' not in script
    assert 'chown' not in script
    cmd = manylinux.docker_command(
        'img',
        mounts={'/work': tmp_path / 'build', '/src': tmp_path / 'src'},
        script='true',
        interactive=True,
        env={'CCACHE_MAXSIZE': '1G', 'PATH': '/bin', 'SCCACHE_DIR': '/x'},
    )
    assert cmd[:4] == ['docker', 'run', '--rm', '-i']
    assert '-t' in cmd
    assert cmd[cmd.index('--volume') + 1] == f'{tmp_path / "build"}:/work'
    assert f'{tmp_path / "src"}:/src' in cmd
    assert cmd.count('--env') == 2
    assert 'PATH' not in cmd
    assert cmd[-4:] == ['img', 'bash', '-c', 'true']
    assert manylinux.file_owner() is None or ':' in manylinux.file_owner()


def test_build_in_manylinux(tmp_path: Path, mocker: MockerFixture):
    mocker.patch.object(tools.shutil, 'which', side_effect=lambda name, **_: f'/usr/bin/{name}')
    mocker.patch.object(manylinux, 'pyvista_source', return_value=tmp_path / 'pv')
    mocker.patch.object(manylinux, 'file_owner', return_value='7:7')
    mocker.patch.object(manylinux.platform, 'machine', return_value='x86_64')
    cfg = _config(
        tmp_path,
        dry_run=False,
        manylinux='img',
        source=tmp_path / 'src',
        cache_dir=tmp_path / 'cc',
        with_=['web'],
        without=['io'],
        jobs=4,
        defines=['X=1'],
        options=['smp=TBB'],
        repair=False,
        version_suffix='dev0+gabc',
    )
    console = Console(record=True, width=4000, force_terminal=False)
    builder = Builder(cfg, console)
    probe = mocker.patch.object(subprocess, 'run', return_value=subprocess.CompletedProcess([], 0))
    run = mocker.patch.object(builder, 'run')
    install = mocker.patch.object(builder, 'phase_install')
    cfg.wheelhouse.mkdir(parents=True)
    (cfg.wheelhouse / 'vtk-9.7.0-cp314-cp314-manylinux_2_28_x86_64.whl').write_text('')
    assert builder.build() == cfg.wheelhouse / 'vtk-9.7.0-cp314-cp314-manylinux_2_28_x86_64.whl'
    assert probe.call_args.args[0] == ['/usr/bin/docker', 'info']
    cmd = run.call_args.args[0]
    assert cmd[:4] == ['docker', 'run', '--rm', '-i']
    assert f'{tmp_path / "vtk-build"}:/work' in cmd
    assert f'{tmp_path / "src"}:/src' in cmd
    assert f'{tmp_path / "cc"}:/cache' in cmd
    assert f'{tmp_path / "pv"}:/pyvista' in cmd
    assert cmd[cmd.index('bash') - 1] == 'img'
    script = cmd[-1]
    assert 'libX11-devel' in script
    assert (
        'pyvista build vtk --build-dir /work --no-install --source /src --cache-dir /cache '
        '--with web --without io --build-type Release --jobs 4 -D X=1 --opt smp=TBB '
        '--no-repair --compiler-cache auto --version-suffix dev0+gabc' in script
    )
    assert 'chown -R 7:7 /work /src /cache' in script
    assert install.called
    assert 'Done' in console.export_text()

    builder = Builder(_config(tmp_path, dry_run=False, manylinux='img'), Console(quiet=True))
    mocker.patch.object(subprocess, 'run', return_value=subprocess.CompletedProcess([], 1))
    with pytest.raises(BuildError, match='daemon is not running'):
        builder.build()
    mocker.patch.object(tools.shutil, 'which', return_value=None)
    with pytest.raises(BuildError, match='docker was not found'):
        builder.build()


def test_build_in_manylinux_dry_run(tmp_path: Path, mocker: MockerFixture):
    mocker.patch.object(tools.shutil, 'which', side_effect=lambda name, **_: f'/usr/bin/{name}')
    probe = mocker.patch.object(subprocess, 'run')
    console = Console(record=True, width=4000, force_terminal=False)
    builder = Builder(_config(tmp_path, manylinux=None, interactive=True), console)
    builder.config.manylinux = 'img'
    mocker.patch.object(builder, 'run')
    assert builder.build() is None
    assert not probe.called
    out = console.export_text()
    assert 'Run the build in the container? (n stops) -> yes (dry run)' in out
    assert not (tmp_path / 'vtk-build').exists()
    builder = Builder(_config(tmp_path, dry_run=False, manylinux='img'), Console(quiet=True))
    mocker.patch.object(subprocess, 'run', return_value=subprocess.CompletedProcess([], 0))
    mocker.patch.object(builder, 'confirm', return_value=False)
    with pytest.raises(BuildError, match='Stopped before the container build'):
        builder.build()


def test_manylinux_without_checkout_or_uid(tmp_path: Path, mocker: MockerFixture):
    fake = tmp_path / 'site-packages' / 'pyvista' / '__init__.py'
    mocker.patch.object(manylinux.pv, '__file__', str(fake))
    assert manylinux.pyvista_source() is None
    mocker.patch.object(manylinux.os, 'getuid', None, create=True)
    assert manylinux.file_owner() is None


def test_msvc_environment_skips_lines_without_value(mocker: MockerFixture, tmp_path: Path):
    mocker.patch.object(
        tools.subprocess,
        'run',
        return_value=subprocess.CompletedProcess('', 0, stdout='banner\nINCLUDE=C:\\inc\n'),
    )
    env = tools.msvc_environment(tmp_path / 'vcvarsall.bat', env={})
    assert env == {'INCLUDE': 'C:\\inc'}


def test_windows_declined_msvc_and_nasm(tmp_path: Path, mocker: MockerFixture):
    mocker.patch.object(tools.shutil, 'which', return_value=None)
    mocker.patch.object(tools, 'find_vcvarsall', return_value=tmp_path / 'vcvarsall.bat')
    activated = mocker.patch.object(tools, 'msvc_environment')
    builder = _builder(tmp_path, mocker, os_name='windows', compiler_cache='none')
    mocker.patch.object(builder, 'confirm', return_value=False)
    builder._activate_msvc(tools.required_tools('windows', builder.manifest.tools, repair=False))
    assert not activated.called
    script = builder.config.src_dir / '.gitlab' / 'ci' / 'download_nasm.cmake'
    script.parent.mkdir(parents=True)
    script.write_text('')
    run = mocker.patch.object(builder, 'run')
    path = builder.env.get('PATH')
    builder._activate_nasm()
    assert not run.called
    assert builder.env.get('PATH') == path


def test_build_in_manylinux_forwarded_args(tmp_path: Path, mocker: MockerFixture):
    mocker.patch.object(tools.shutil, 'which', side_effect=lambda name, **_: f'/usr/bin/{name}')
    mocker.patch.object(manylinux, 'pyvista_source', return_value=None)
    mocker.patch.object(subprocess, 'run', return_value=subprocess.CompletedProcess([], 0))
    cfg = _config(tmp_path, dry_run=False, manylinux='img', url='https://x/vtk.git', ref='v1')
    builder = Builder(cfg, Console(quiet=True))
    run = mocker.patch.object(builder, 'run')
    mocker.patch.object(builder, 'phase_install')
    cfg.wheelhouse.mkdir(parents=True)
    (cfg.wheelhouse / 'vtk-1-cp314-cp314-manylinux_2_28_x86_64.whl').write_text('')
    builder.build()
    cmd = run.call_args.args[0]
    assert not any(part.endswith(':/pyvista') for part in cmd)
    assert '--url https://x/vtk.git --ref v1' in cmd[-1]
    cfg.all_ = True
    assert '--all' in builder.container_args()
    cfg.all_, cfg.minimal = False, True
    assert '--minimal' in builder.container_args()
    mocker.patch.object(phases, 'find_manifest', side_effect=ManifestError('bad manifest'))
    with pytest.raises(BuildError, match='bad manifest'):
        builder.build()


def test_compiler_cache_row(tmp_path: Path, mocker: MockerFixture):
    builder = _builder(tmp_path, mocker, compiler_cache='none')
    assert builder._compiler_cache_row() == (
        'compiler cache',
        '-',
        '[dim]off (--compiler-cache none)[/dim]',
    )
    mocker.patch.object(tools.shutil, 'which', return_value=None)
    builder.config.compiler_cache = 'auto'
    assert builder._compiler_cache_row()[0] == 'ccache/sccache (compiler cache)'
    assert 'optional' in builder._compiler_cache_row()[2]
    builder.config.compiler_cache = 'sccache'
    assert builder._compiler_cache_row() == ('sccache (compiler cache)', '-', '[red]missing[/red]')
    mocker.patch.object(tools.shutil, 'which', side_effect=lambda name, **_: f'/usr/bin/{name}')
    builder.config.compiler_cache = 'auto'
    assert builder._compiler_cache_row() == (
        'ccache (compiler cache)',
        '/usr/bin/ccache',
        '[green]ok[/green]',
    )
