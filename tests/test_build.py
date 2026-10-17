"""Tests for the backend build engine and the manifest machinery behind it."""

from __future__ import annotations

import re
import subprocess
from typing import TYPE_CHECKING

import pytest
from rich.console import Console

from pyvista._build import BuildConfig
from pyvista._build import Builder
from pyvista._build import BuildError
from pyvista._build import ManifestError
from pyvista._build import load_manifest
from pyvista._build import phases
from pyvista._build import tools
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
    console = Console(record=True, width=200, force_terminal=False)
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
    console = Console(record=True, width=200, force_terminal=False)
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
    console = Console(record=True, width=200, force_terminal=False)
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
    console = Console(record=True, width=200, force_terminal=False)
    Builder(_config(tmp_path), console).build()
    out = console.export_text()
    assert 'Missing Python packages: cmake>=3.22, auditwheel, wheel' in out
    assert f'$ {phases._join(["pip", "install", "cmake>=3.22", "auditwheel", "wheel"])}' in out


def test_too_old_tool(tmp_path: Path, mocker: MockerFixture):
    mocker.patch.object(tools.shutil, 'which', side_effect=lambda name, **_: f'/usr/bin/{name}')
    mocker.patch.object(tools, 'executable_version', return_value='3.10')
    mocker.patch.object(tools, 'python_build_packages', return_value=[])
    mocker.patch.object(
        tools, 'pip_install_command', side_effect=lambda *s, **_: ['pip', 'install', *s]
    )
    console = Console(record=True, width=200, force_terminal=False)
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
    console = Console(record=True, width=200, force_terminal=False)
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
