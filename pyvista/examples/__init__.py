"""Examples module."""

from __future__ import annotations

import importlib
import inspect
from typing import TYPE_CHECKING

from . import cells as cells
from . import downloads as downloads
from . import examples as examples
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


def _public_functions(module: Any) -> list[str]:
    """Return the names of ``module``'s public, module-defined functions.

    This is used to build ``__all__`` below so that the ``load_*`` and
    ``download_*`` functions re-exported here via ``from .examples import
    *`` and ``from .downloads import *`` are documented (and cross-
    referenceable, e.g. with ``:func:`~pyvista.examples.load_uniform```) at
    their public ``pyvista.examples`` location instead of only at their
    private ``pyvista.examples.examples``/``pyvista.examples.downloads``
    submodule location.

    """
    return sorted(
        name
        for name, obj in vars(module).items()
        if not name.startswith('_')
        and inspect.isfunction(obj)
        and obj.__module__ == module.__name__
    )


__all__ = [
    'cells',
    'downloads',
    'examples',
    'planets',
    'generate_cell_blocks',
    'plot_cell',
    *_public_functions(downloads),
    *_public_functions(examples),
]
