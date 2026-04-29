from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type checkers cannot resolve the dynamic lazy vtk imports,
    # so we import everything when type-checking
    from pyvista.core._vtk_core import *

_THIS_MODULE = sys.modules[__name__]

# Magic imports needed to make LaTeX rendering work. See https://discourse.vtk.org/t/how-to-check-if-mathtext-is-supported-without-importing-all-of-vtk/16038
# isort: off
import vtkmodules.vtkRenderingFreeType  # noqa: F401
import vtkmodules.vtkRenderingMatplotlib  # noqa: F401
# isort: on


from pyvista.core import _vtk_core


def __getattr__(name: str):
    """Fallback to import from vtk core."""
    try:
        obj = getattr(_vtk_core, name)
    except AttributeError:
        msg = f'module {__name__!r} has no attribute {name!r}'
        raise AttributeError(msg) from None

    # Cache object for next access
    _THIS_MODULE.__dict__[name] = obj
    return obj
