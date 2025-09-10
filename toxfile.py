from __future__ import annotations  # noqa: D100

import os
from typing import TYPE_CHECKING

from tox.plugin import impl

if TYPE_CHECKING:
    from tox.config.sets import EnvConfigSet
    from tox.session.state import State


@impl
def tox_add_env_config(env_conf: EnvConfigSet, state: State) -> None:  # noqa: ARG001, D103
    if env_conf.env_name != 'docs-build':
        return

    if os.environ.get('CI', 'false').lower() == 'true':
        env_conf['set_env'].update({'VTK_DEFAULT_OPENGL_WINDOW': 'vtkEGLRenderWindow'})
