"""Examples module."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from . import planets as planets
from .cells import generate_cell_blocks as generate_cell_blocks
from .cells import plot_cell as plot_cell
from .downloads import *
from .examples import *

if TYPE_CHECKING:
    from typing import Any

_DEPRECATED_SUBMODULES = {'vrml', 'download_3ds', 'gltf'}


def __getattr__(name: str) -> Any:
    if name in _DEPRECATED_SUBMODULES:  # pragma: no cover
        module = importlib.import_module(f'.{name}', __name__)
        globals()[name] = module  # cache on the package so this only runs once
        return module
    msg = f'module {__name__!r} has no attribute {name!r}'
    raise AttributeError(msg)
