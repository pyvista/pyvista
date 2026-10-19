"""Tests for build tool detection and the default job count."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import TYPE_CHECKING

from pyvista._build import tools
from pyvista._build.manifest import OS_NAMES

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def test_pip_install_command(mocker: MockerFixture):
    mocker.patch.object(tools.importlib.util, 'find_spec', return_value=object())
    cmd = tools.pip_install_command('cmake', args=('--no-deps',))
    assert cmd[1:] == ['-m', 'pip', 'install', '--no-deps', 'cmake']
    mocker.patch.object(tools.importlib.util, 'find_spec', return_value=None)
    mocker.patch.object(tools.shutil, 'which', return_value=None)
    assert tools.pip_install_command('cmake') is None


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
    hints = {
        (os_name, name): tools.compiler_cache_instructions(os_name, name)
        for os_name in OS_NAMES
        for name in ('auto', 'ccache', 'sccache')
    }
    assert hints['linux', 'sccache'] == 'pip install sccache'
    assert hints['macos', 'sccache'] == 'pip install sccache   (or: brew install sccache)'
    assert hints['windows', 'sccache'] == 'pip install sccache   (or: choco install sccache)'
    assert hints['windows', 'ccache'] == 'choco install ccache'
    assert hints['macos', 'auto'] == 'brew install ccache'
    assert 'sccache' in hints['windows', 'auto']
    assert all(tools.compiler_instructions(os_name) for os_name in OS_NAMES)
    tool = tools.Tool('t', ('t',), spec='>=1')
    mocker.patch.object(tools.shutil, 'which', return_value='/usr/bin/t')
    mocker.patch.object(tools, 'executable_version', return_value=None)
    assert tools.detect(tool)
    mocker.patch.object(tools, 'executable_version', return_value='not.a.version')
    assert tools.detect(tool)


def test_installer_and_build_packages(mocker: MockerFixture):
    mocker.patch.object(tools.importlib.util, 'find_spec', return_value=None)
    assert tools.python_build_packages() == ['setuptools', 'wheel']
    mocker.patch.object(tools.shutil, 'which', return_value='/opt/uv')
    assert tools.pip_install_command('x', args=('--no-deps',)) == [
        '/opt/uv',
        'pip',
        'install',
        '--python',
        sys.executable,
        '--no-deps',
        'x',
    ]
    mocker.patch.object(
        tools.importlib.util, 'find_spec', side_effect=lambda n: None if n == 'wheel' else object()
    )
    assert tools.python_build_packages() == ['wheel']


def test_physical_memory(mocker: MockerFixture):
    """Physical memory comes from sysconf, or GlobalMemoryStatusEx on Windows."""
    mocker.stopall()
    assert tools.physical_memory() > 0
    mocker.patch.object(tools.os, 'sysconf', side_effect=ValueError, create=True)
    mocker.patch.object(tools.sys, 'platform', 'linux')
    assert tools.physical_memory() is None
    import ctypes

    def status(ref):
        ref._obj.ullTotalPhys = 16 * 2**30
        return 1

    kernel32 = mocker.Mock(**{'GlobalMemoryStatusEx.side_effect': status})
    mocker.patch.object(ctypes, 'windll', mocker.Mock(kernel32=kernel32), create=True)
    mocker.patch.object(tools.sys, 'platform', 'win32')
    assert tools.physical_memory() == 16 * 2**30
    kernel32.GlobalMemoryStatusEx.side_effect = None
    kernel32.GlobalMemoryStatusEx.return_value = 0
    assert tools.physical_memory() is None


def test_required_tools_per_os():
    """Each OS gets git, cmake, ninja, its compiler and, with repair, its repair tools."""
    linux = tools.required_tools('linux', {'cmake': '>=3.22'}, repair=True)
    assert [t.name for t in linux] == [
        'git',
        'cmake',
        'ninja',
        'c++ compiler',
        'auditwheel',
        'patchelf',
    ]
    assert [t.label for t in linux[1:3]] == ['cmake>=3.22', 'ninja']
    assert linux[3].check == ()
    assert linux[0].instructions.startswith('Install git')
    assert [(t.name, t.pip) for t in linux[4:]] == [
        ('auditwheel', 'auditwheel'),
        ('patchelf', 'patchelf'),
    ]
    macos = tools.required_tools('macos', {}, repair=True)
    assert macos[3].check == ('xcode-select', '-p')
    assert [(t.name, t.pip) for t in macos[4:]] == [('delocate-wheel', 'delocate')]
    windows = tools.required_tools('windows', {}, repair=False)
    assert [t.name for t in windows] == ['git', 'cmake', 'ninja', 'msvc']
    assert windows[3].executables == ('cl',)
    windows = tools.required_tools('windows', {}, repair=True)
    assert [(t.name, t.pip) for t in windows[4:]] == [('delvewheel', 'delvewheel')]


def test_detect_check_and_version(mocker: MockerFixture):
    """A failing check command or a too-old version counts as missing."""
    which = mocker.patch.object(tools.shutil, 'which', return_value='/usr/bin/c++')
    succeeds = mocker.patch.object(tools, 'command_succeeds', return_value=False)
    compiler = tools.Tool('c++ compiler', ('c++',), check=('xcode-select', '-p'))
    assert not tools.detect(compiler)
    assert compiler.path is None
    succeeds.return_value = True
    assert tools.detect(compiler)
    assert compiler.path == '/usr/bin/c++'
    which.return_value = None
    assert not tools.detect(tools.Tool('x', ('x',)))
    which.return_value = '/usr/bin/cmake'
    version = mocker.patch.object(tools, 'executable_version', return_value='3.20.0')
    cmake = tools.Tool('cmake', ('cmake',), spec='>=3.22')
    assert not tools.detect(cmake)
    assert cmake.version == '3.20.0'
    version.return_value = '3.30.1'
    assert tools.detect(cmake)


def test_which_and_command_succeeds(tmp_path: Path):
    """Executables resolve against a given PATH, and commands report their exit status."""
    assert tools.which('no-such-tool-here', {'PATH': str(tmp_path)}) == 'no-such-tool-here'
    exe = tmp_path / ('fake-tool.exe' if os.name == 'nt' else 'fake-tool')
    exe.write_text('')
    exe.chmod(0o755)
    assert Path(tools.which('fake-tool', {'PATH': str(tmp_path)})) == exe
    assert tools.which('fake-tool', {}) == 'fake-tool'
    assert tools.which(sys.executable) == sys.executable
    assert tools.scripts_dir()
    assert tools.command_succeeds([sys.executable, '-c', 'pass'])
    assert not tools.command_succeeds([sys.executable, '-c', 'raise SystemExit(3)'])
    assert not tools.command_succeeds([str(tmp_path / 'missing')])


def test_default_jobs():
    """The default job count is one per core, at most one per 2 GB of memory."""
    gib = 2**30
    assert tools.default_jobs(8, None) == 8
    assert tools.default_jobs(0, None) == 1
    assert tools.default_jobs(8, 64 * gib) == 8
    assert tools.default_jobs(16, 24 * gib) == 12
    assert tools.default_jobs(8, gib) == 1
