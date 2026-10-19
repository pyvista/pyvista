"""Detection and installation of the tools a backend build needs."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
from typing import TYPE_CHECKING

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion
from packaging.version import Version

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence


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
    check: tuple[str, ...] = ()

    @property
    def label(self) -> str:  # numpydoc ignore=RT01
        """Name plus version requirement, for messages."""
        return f'{self.name}{self.spec}' if self.spec else self.name


_VERSION_RE = re.compile(r'(\d+\.\d+(?:\.\d+)?)')


def git_tool() -> Tool:
    """Return the git :class:`Tool`.

    Returns
    -------
    Tool
        Fresh, undetected git tool.

    """
    return Tool('git', ('git',), instructions='Install git: https://git-scm.com/downloads')


def scripts_dir() -> str:
    """Return the directory pip installs this interpreter's console scripts into.

    Returns
    -------
    str
        Scripts directory.

    """
    return sysconfig.get_path('scripts')


def which(name: str, env: Mapping[str, str] | None = None) -> str:
    """Resolve an executable against an environment's ``PATH``.

    Parameters
    ----------
    name : str
        Executable name or path.

    env : Mapping[str, str], optional
        Environment whose ``PATH`` to search. Defaults to the process environment.

    Returns
    -------
    str
        Full path when found, else ``name`` unchanged.

    """
    return shutil.which(name, path=(os.environ if env is None else env).get('PATH')) or name


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


def command_succeeds(cmd: Sequence[str]) -> bool:
    """Return whether a command runs and exits with status 0.

    Parameters
    ----------
    cmd : Sequence[str]
        Command and arguments.

    Returns
    -------
    bool
        ``True`` on exit status 0, ``False`` on any other status or when it cannot run.

    """
    try:
        result = subprocess.run(list(cmd), capture_output=True, check=False, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def detect(tool: Tool, env: Mapping[str, str] | None = None) -> bool:
    """Look a tool up on ``PATH`` and record what was found on it.

    Parameters
    ----------
    tool : Tool
        Tool to look for; its ``path`` and ``version`` are updated. A tool whose
        ``check`` command fails counts as missing.

    env : Mapping[str, str], optional
        Environment whose ``PATH`` to search. Defaults to the process environment.

    Returns
    -------
    bool
        Whether the tool is present and satisfies its version requirement.

    """
    path_var = (os.environ if env is None else env).get('PATH')
    tool.path = next(
        (p for name in tool.executables if (p := shutil.which(name, path=path_var))), None
    )
    tool.version = None
    if tool.path is not None and tool.check and not command_succeeds(tool.check):
        tool.path = None
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
        'sudo apt install build-essential libgl1-mesa-dev libx11-dev libxcursor-dev '
        'or sudo dnf install gcc-c++ mesa-libGL-devel libX11-devel libXcursor-devel'
    )


#: Wheel repair executable on each operating system.
REPAIRERS = {'linux': 'auditwheel', 'macos': 'delocate-wheel', 'windows': 'delvewheel'}


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
        git_tool(),
        Tool('cmake', ('cmake',), spec=tool_specs.get('cmake', ''), pip='cmake'),
        Tool('ninja', ('ninja',), spec=tool_specs.get('ninja', ''), pip='ninja'),
    ]
    if os_name == 'windows':
        tools.append(Tool('msvc', ('cl',), instructions=compiler_instructions(os_name)))
    else:
        compilers = ('c++', 'clang++', 'g++')
        # the macOS /usr/bin/c++ shim exists even without the Command Line Tools
        check = ('xcode-select', '-p') if os_name == 'macos' else ()
        instructions = compiler_instructions(os_name)
        tools.append(Tool('c++ compiler', compilers, instructions=instructions, check=check))
    if repair:
        pip = {'linux': 'auditwheel', 'macos': 'delocate', 'windows': 'delvewheel'}
        repairer = REPAIRERS[os_name]
        tools.append(Tool(repairer, (repairer,), pip=pip[os_name]))
        if os_name == 'linux':
            tools.append(Tool('patchelf', ('patchelf',), pip='patchelf'))
    return tools


PYTHON_BUILD_PACKAGES = ('setuptools', 'wheel')

MEMORY_PER_JOB = 2 * 1024**3


def physical_memory() -> int | None:
    """Return the machine's physical memory in bytes, or ``None`` when it cannot be read.

    Returns
    -------
    int | None
        Total physical memory in bytes.

    """
    if sys.platform == 'win32':

        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ('dwLength', ctypes.c_ulong),
                ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong),
                ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong),
                ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong),
                ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
            ]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ullTotalPhys)
    try:
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except (AttributeError, OSError, ValueError):
        return None


def default_jobs(cores: int, memory: int | None) -> int:
    """Return the parallel compile jobs: one per core, at most one per 2 GB of memory.

    Parameters
    ----------
    cores : int
        CPU cores.

    memory : int | None
        Physical memory in bytes, or ``None`` when unknown.

    Returns
    -------
    int
        Job count, at least 1.

    """
    if memory is None:
        return max(1, cores)
    return max(1, min(cores, memory // MEMORY_PER_JOB))


def python_build_packages() -> list[str]:
    """Return the packages ``setup.py bdist_wheel`` needs that are not importable.

    Returns
    -------
    list[str]
        Missing package names among ``setuptools`` and ``wheel``.

    """
    return [name for name in PYTHON_BUILD_PACKAGES if importlib.util.find_spec(name) is None]
