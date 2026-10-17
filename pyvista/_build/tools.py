"""Detection and installation of the tools a backend build needs."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
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


PYTHON_BUILD_PACKAGES = ('setuptools', 'wheel')


def python_build_packages() -> list[str]:
    """Return the packages ``setup.py bdist_wheel`` needs that are not importable.

    Returns
    -------
    list[str]
        Missing package names among ``setuptools`` and ``wheel``.

    """
    return [name for name in PYTHON_BUILD_PACKAGES if importlib.util.find_spec(name) is None]
