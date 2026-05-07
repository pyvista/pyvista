from __future__ import annotations  # noqa: D100

import logging
import os
import re
from typing import TYPE_CHECKING

from packaging.version import Version
from tox.execute.api import StdinSource
from tox.plugin import impl
from tox.tox_env.errors import Fail
from tox_uv._run import UvVenvRunner

if TYPE_CHECKING:
    from tox.config.sets import EnvConfigSet
    from tox.session.state import State
    from tox.tox_env.api import ToxEnv
    from tox.tox_env.python.package import WheelPackage
    from tox.tox_env.python.pip.req_file import PythonDeps
    from tox_uv._installer import UvInstaller


@impl
def tox_add_env_config(env_conf: EnvConfigSet, state: State) -> None:  # noqa: ARG001, D103
    if os.environ.get('CI', 'false').lower() != 'true':
        return

    if env_conf.env_name in ['docs-build', 'doctest-modules']:
        env_conf['set_env'].update({'VTK_DEFAULT_OPENGL_WINDOW': 'vtkEGLRenderWindow'})
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
        val = '-n4' if vtk_version == 'latest' or Version(vtk_version) >= Version('9.4.0') else ''

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
    constraints_file = tox_env.env_dir / 'constraints.txt'

    if of_type == 'deps' and isinstance(tox_env, UvVenvRunner):
        constraints_file.write_text('\n'.join(arguments.lines()))

    if of_type in 'package':
        _arguments: list[WheelPackage] = arguments
        constraints_file_dep = f'-c {constraints_file}'
        for package in _arguments:
            getattr(package, 'deps', []).append(constraints_file_dep)


@impl
def tox_before_run_commands(tox_env: ToxEnv) -> None:
    """Check vtk deps.

    When specifying the vtk version in the env name, check the installed vtk version with
    a `freeze` command.
    """
    reg = re.compile(r'.*vtk_(\d+\.\d+\.\d+).*')
    if not (m := re.match(reg, tox_env.name)):
        return

    required = m.group(1)

    installer: UvInstaller = tox_env.installer
    out = tox_env.execute(
        installer.freeze_cmd(),
        stdin=StdinSource.OFF,
        show=False,
        run_id='check_vtk_deps',
    )
    matches = re.findall(r'\nvtk==(\d+\.\d+\.\d+)\n', out.out, re.MULTILINE)
    if len(matches) != 1:
        msg = 'Could not find the installed vtk version in the output of `uv pip freeze`'
        raise Fail(msg)

    installed = matches[0]

    if installed != required:
        msg = (
            f'The installed vtk version ({installed}) does not match the required version',
            f' ({required}).',
        )
        raise Fail(msg)

    logging.warning(  # noqa: LOG015
        f'Installed vtk version ({installed}) matches the required version ({required}).'
    )
