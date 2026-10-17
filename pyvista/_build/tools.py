"""Detection and installation of the tools a backend build needs."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion
from packaging.version import Version

if TYPE_CHECKING:
    from collections.abc import Mapping


def host_os() -> str:
    """Return the operating system name used by manifests.

    Returns
    -------
    str
        ``'linux'``, ``'macos'`` or ``'windows'``.

    """
    system = platform.system()
    if system == 'Darwin':
        return 'macos'
    if system == 'Windows':
        return 'windows'
    return 'linux'


@dataclass
class Tool:
    """One executable the build needs, and how to get it when it is missing."""

    name: str
    executables: tuple[str, ...]
    spec: str = ''
    pip: str | None = None
    instructions: str = ''
    path: str | None = None
    version: str | None = None

    @property
    def label(self) -> str:  # numpydoc ignore=RT01
        """Name plus version requirement, for messages."""
        return f'{self.name}{self.spec}' if self.spec else self.name


_VERSION_RE = re.compile(r'(\d+\.\d+(?:\.\d+)?)')


def executable_version(path: str) -> str | None:
    """Return the version an executable reports.

    Parameters
    ----------
    path : str
        Executable to run with ``--version``.

    Returns
    -------
    str | None
        First ``x.y[.z]`` in the output, or ``None`` when it cannot be read.

    """
    try:
        out = subprocess.run(
            [path, '--version'], capture_output=True, text=True, check=False, timeout=30
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = _VERSION_RE.search(out)
    return match.group(1) if match else None


def detect(tool: Tool, env: Mapping[str, str] | None = None) -> bool:
    """Look a tool up on ``PATH`` and record what was found on it.

    Parameters
    ----------
    tool : Tool
        Tool to look for; its ``path`` and ``version`` are updated.

    env : Mapping[str, str], optional
        Environment whose ``PATH`` to search. Defaults to the process environment.

    Returns
    -------
    bool
        Whether the tool is present and satisfies its version requirement.

    """
    path_var = (env or os.environ).get('PATH')
    tool.path = next(
        (p for name in tool.executables if (p := shutil.which(name, path=path_var))), None
    )
    tool.version = None
    if tool.path is None:
        return False
    if not tool.spec:
        return True
    tool.version = executable_version(tool.path)
    if tool.version is None:
        return True
    try:
        return Version(tool.version) in SpecifierSet(tool.spec)
    except InvalidVersion:
        return True


def pip_install_command(*specs: str, args: tuple[str, ...] = ()) -> list[str] | None:
    """Return the command that installs packages into the running interpreter.

    Parameters
    ----------
    *specs : str
        Requirement specifiers or wheel paths.

    args : tuple[str, ...], default: ()
        Extra installer arguments placed before the specifiers.

    Returns
    -------
    list[str] | None
        ``pip`` or ``uv pip`` command, or ``None`` when neither is available.

    """
    python = sys.executable
    if importlib.util.find_spec('pip') is not None:
        return [python, '-m', 'pip', 'install', *args, *specs]
    uv = shutil.which('uv')
    if uv is not None:
        return [uv, 'pip', 'install', '--python', python, *args, *specs]
    return None


def compiler_instructions(os_name: str) -> str:
    """Return how to get a C++ compiler.

    Parameters
    ----------
    os_name : str
        Operating system name.

    Returns
    -------
    str
        Installation instructions.

    """
    if os_name == 'macos':
        return 'Install the Xcode Command Line Tools: xcode-select --install'
    if os_name == 'windows':
        return (
            'Install Visual Studio Build Tools with the "Desktop development with C++" '
            'workload: https://visualstudio.microsoft.com/visual-cpp-build-tools/'
        )
    return (
        'Install a C++ compiler and the OpenGL headers, for example '
        'apt install build-essential libgl1-mesa-dev libx11-dev libxcursor-dev '
        'or dnf install gcc-c++ mesa-libGL-devel libX11-devel libXcursor-devel'
    )


def required_tools(os_name: str, tool_specs: Mapping[str, str], *, repair: bool) -> list[Tool]:
    """Return the tools a build needs on one operating system.

    Parameters
    ----------
    os_name : str
        Operating system name.

    tool_specs : Mapping[str, str]
        Version requirements from the manifest's ``[tools]`` table.

    repair : bool
        Whether the wheel repair tool is needed.

    Returns
    -------
    list[Tool]
        Tools to detect, in the order they are reported.

    """
    tools = [
        Tool('git', ('git',), instructions='Install git: https://git-scm.com/downloads'),
        Tool('cmake', ('cmake',), spec=tool_specs.get('cmake', ''), pip='cmake'),
        Tool('ninja', ('ninja',), spec=tool_specs.get('ninja', ''), pip='ninja'),
    ]
    if os_name == 'windows':
        tools.append(Tool('msvc', ('cl',), instructions=compiler_instructions(os_name)))
    else:
        compilers = ('c++', 'clang++', 'g++')
        tools.append(Tool('c++ compiler', compilers, instructions=compiler_instructions(os_name)))
    if repair:
        repairer = {'linux': 'auditwheel', 'macos': 'delocate-wheel', 'windows': 'delvewheel'}
        pip = {'linux': 'auditwheel', 'macos': 'delocate', 'windows': 'delvewheel'}
        tools.append(Tool(repairer[os_name], (repairer[os_name],), pip=pip[os_name]))
    return tools


def python_build_packages() -> list[str]:
    """Return the packages ``setup.py bdist_wheel`` needs that are not importable.

    Returns
    -------
    list[str]
        Missing package names among ``setuptools`` and ``wheel``.

    """
    return [name for name in ('setuptools', 'wheel') if importlib.util.find_spec(name) is None]


def compiler_cache_tool(
    choice: str, os_name: str, env: Mapping[str, str] | None = None
) -> Tool | None:
    """Resolve the ``--compiler-cache`` choice to a launcher found on ``PATH``.

    Parameters
    ----------
    choice : str
        ``'auto'``, ``'ccache'``, ``'sccache'`` or ``'none'``.

    os_name : str
        Operating system name; ``auto`` prefers ``sccache`` on Windows.

    env : Mapping[str, str], optional
        Environment whose ``PATH`` to search.

    Returns
    -------
    Tool | None
        Detected launcher, or ``None`` when none is available.

    """
    if choice == 'none':
        return None
    order: tuple[str, ...] = (choice,)
    if choice == 'auto':
        order = ('sccache', 'ccache') if os_name == 'windows' else ('ccache', 'sccache')
    for name in order:
        tool = Tool(name, (name,))
        if detect(tool, env):
            return tool
    return None


def compiler_cache_instructions(os_name: str) -> str:
    """Return how to install a compiler cache.

    Parameters
    ----------
    os_name : str
        Operating system name.

    Returns
    -------
    str
        Installation instructions.

    """
    if os_name == 'macos':
        return 'brew install ccache'
    if os_name == 'windows':
        return 'pip install sccache   (or: choco install sccache)'
    return 'apt install ccache   (or: dnf install ccache)'


def vswhere_path() -> Path | None:
    """Return the path to ``vswhere.exe`` when Visual Studio's installer is present.

    Returns
    -------
    Path | None
        Path to ``vswhere.exe``, or ``None``.

    """
    base = os.environ.get('PROGRAMFILES(X86)') or os.environ.get('PROGRAMFILES')
    if not base:
        return None
    path = Path(base) / 'Microsoft Visual Studio' / 'Installer' / 'vswhere.exe'
    return path if path.is_file() else None


def find_vcvarsall() -> Path | None:
    """Locate ``vcvarsall.bat`` through ``vswhere``.

    Returns
    -------
    Path | None
        Path to ``vcvarsall.bat``, or ``None`` when no suitable Visual Studio is installed.

    """
    vswhere = vswhere_path()
    if vswhere is None:
        return None
    out = subprocess.run(
        [
            str(vswhere),
            '-latest',
            '-products',
            '*',
            '-requires',
            'Microsoft.VisualStudio.Component.VC.Tools.x86.x64',
            '-property',
            'installationPath',
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if not out:
        return None
    path = Path(out) / 'VC' / 'Auxiliary' / 'Build' / 'vcvarsall.bat'
    return path if path.is_file() else None


def msvc_environment(
    vcvarsall: Path, arch: str = 'x64', env: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Return the environment after running ``vcvarsall.bat``.

    Parameters
    ----------
    vcvarsall : Path
        ``vcvarsall.bat`` to run.

    arch : str, default: 'x64'
        Target architecture argument.

    env : Mapping[str, str], optional
        Environment to run it in and extend. Defaults to the process environment.

    Returns
    -------
    dict[str, str]
        ``env`` plus the additions from Visual Studio.

    """
    base = dict(env if env is not None else os.environ)
    out = subprocess.run(  # noqa: S602
        f'"{vcvarsall}" {arch} >nul && set',
        shell=True,
        env=base,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for line in out.splitlines():
        key, sep, value = line.partition('=')
        if sep:
            base[key] = value
    return base
