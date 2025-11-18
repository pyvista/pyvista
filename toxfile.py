from __future__ import annotations  # noqa: D100

import os
import re
from typing import TYPE_CHECKING

from packaging.version import Version
from tox.plugin import impl

if TYPE_CHECKING:
    from tox.config.sets import EnvConfigSet
    from tox.session.state import State


@impl
def tox_add_env_config(env_conf: EnvConfigSet, state: State) -> None:  # noqa: ARG001, D103
    if 'vtk_dev' not in env_conf.env_name:
        # Remove vtk from deps since it's installed separately
        deps = env_conf.get('deps', [])
        filtered = [d for d in deps if not d.raw.strip().startswith('vtk')]
        env_conf['deps'] = filtered

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
