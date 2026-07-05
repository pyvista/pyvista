from __future__ import annotations  # noqa: D100

import logging
import os
import re
from typing import TYPE_CHECKING

from packaging.requirements import Requirement
from packaging.version import Version
from tox.execute.api import StdinSource
from tox.plugin import impl
from tox.tox_env.errors import Fail
from tox_uv._run import UvVenvRunner

if TYPE_CHECKING:
    from collections.abc import Generator

    from tox.config.sets import EnvConfigSet
    from tox.session.state import State
    from tox.tox_env.api import ToxEnv
    from tox.tox_env.python.package import WheelPackage
    from tox.tox_env.python.pip.req_file import PythonDeps
    from tox_uv._installer import UvInstaller

CONSTRAINTS_FILE = 'constraints.txt'


def _get_ci_xdist_auto_workers() -> str:
    """Return a resource-aware worker count for heavy CI xdist jobs."""
    cpu_count = os.cpu_count() or 1
    return str(max(1, cpu_count // 2))


@impl
def tox_add_env_config(env_conf: EnvConfigSet, state: State) -> None:  # noqa: ARG001, D103
    if os.environ.get('CI', 'false').lower() != 'true':
        return

    if env_conf.env_name == 'docs-build':
        env_conf['set_env'].update({'VTK_DEFAULT_OPENGL_WINDOW': 'vtkEGLRenderWindow'})
        return

    if env_conf.env_name == 'doctest-modules':
        updated = {'VTK_DEFAULT_OPENGL_WINDOW': 'vtkEGLRenderWindow'}
        if os.environ['RUNNER_OS'] == 'Linux':
            updated['PYTEST_XDIST_AUTO_NUM_WORKERS'] = _get_ci_xdist_auto_workers()
        env_conf['set_env'].update(updated)
        return

    # For plotting tests on Linux with vtk < 9.4, some segfaults have been spotted
    # See https://github.com/pyvista/pyvista/pull/7885#issuecomment-3263323390
    # Therefore, the "PARALLEL" env variable is updated
    if ('plotting' in env_conf.name) and os.environ['RUNNER_OS'] == 'Linux':
        # Test the vtk version from TOX_FACTOR env variable
        reg = re.compile('.*-vtk_(.*)$')
        if not (factor := os.environ.get('TOX_FACTOR')):
            return

        if not (m := re.match(reg, factor)):
            msg = 'The `TOX_FACTOR` env variable is malformed. Could not get the vtk version.'
            raise ValueError(msg)

        vtk_version = m.group(1)
        # Non-numeric factors ("latest", "cvista") track a recent VTK (>= 9.4).
        recent = vtk_version in ('latest', 'cvista') or Version(vtk_version) >= Version('9.4.0')
        val = '-n4' if recent else ''

        updated = {'PARALLEL': val}
        env_conf['set_env'].update(updated)


@impl
def tox_on_install(  # noqa: PLR0917
    tox_env: ToxEnv,
    arguments: list[WheelPackage] | PythonDeps,
    section: str,  # noqa: ARG001
    of_type: str,
) -> None:
    """Before installing:
    * save environment `deps` to a constraints file
    * apply constraints file during `install_package_deps` step.

    This is needed since dependencies installed in subsequent steps (eg. `dependency-groups`) may
    override the `deps` ones.
    See https://github.com/pyvista/pyvista/issues/8635.

    Mostly inspired by https://github.com/tox-dev/tox/issues/2386#issuecomment-1396105380
    """  # noqa: D205
    global CONSTRAINTS_FILE  # noqa: PLW0603
    CONSTRAINTS_FILE = tox_env.env_dir / CONSTRAINTS_FILE

    if of_type == 'deps' and isinstance(tox_env, UvVenvRunner):
        CONSTRAINTS_FILE.write_text('\n'.join(arguments.lines()))

    if of_type in 'package':
        _arguments: list[WheelPackage] = arguments
        constraints_file_dep = f'-c {CONSTRAINTS_FILE}'
        for package in _arguments:
            getattr(package, 'deps', []).append(constraints_file_dep)


def _normalize_package_name(name: str) -> str:
    """Taken from https://packaging.python.org/en/latest/specifications/name-normalization/#name-normalization."""
    return re.sub(r'[-_.]+', '-', name).lower()


def _get_freezed_requirements(lines: list[str]) -> Generator[tuple[str, Version], None, None]:
    """From a freeze output, get the list of installed requirements as (name, version) tuples.

    Note that the name is normalized per packaging specifications (see https://packaging.python.org/en/latest/specifications/name-normalization/#name-normalization).
    """
    for l in lines:
        if l.startswith('-e '):  # installed in editable mode, e.g., MNE-Python
            continue
        req = Requirement(l)
        if (m := re.match(r'(.*)==(\S+)', str(req.specifier))) is not None:
            yield _normalize_package_name(req.name), Version(m.group(2))


@impl
def tox_before_run_commands(tox_env: ToxEnv) -> None:
    """Check that deps declared in the constraints_file (ie. during the `deps` step above)
    are indeed installed in the environment before running the tests using a freeze command.
    """  # noqa: D205
    # Load requirements from the constraints file
    with CONSTRAINTS_FILE.open() as f:
        requirements = [Requirement(l) for l in f.read().splitlines()]

    # Load installed deps using freeze
    installer: UvInstaller = tox_env.installer
    cmd = [*installer.freeze_cmd(), '--exclude-editable']
    out = tox_env.execute(
        cmd=cmd,
        stdin=StdinSource.OFF,
        show=False,
        run_id='check_deps',
    )

    # Parse installed deps from the freeze output.
    # Relies on the fact the the freeze command outputs lines in the
    # form of `package==version` (eg. `vtk==9.2.2`)
    installed = list(_get_freezed_requirements(out.out.splitlines()))

    # Check that the installed requirements match the constraints file ones
    for req in requirements:
        # Get the installed requirement matching the current one
        name = _normalize_package_name(req.name)

        installed_req = next((r for r in installed if r[0] == name), None)
        if not installed_req:
            msg = f'The required package {name} is not installed in the environment.'
            raise Fail(msg)

        # Check that the installed requirement version matches the specifier in
        # the constraints file
        if (version_installed := installed_req[1]) not in req.specifier:
            msg = (
                f'The installed version of {req.name} ({version_installed}) does not match',
                f' the required version ({req.specifier}).',
            )
            raise Fail(msg)

        logging.warning(  # noqa: LOG015
            f'Installed {name} version ({version_installed}) matches the required version'
            f' ({req.specifier}).'
        )
