"""Type aliases for annotating code that uses PyVista."""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING

from pyvista._version import _is_deprecation_due
from pyvista._warn_external import warn_external
from pyvista.core._typing_core import ArrayLikeBool as ArrayLikeBool
from pyvista.core._typing_core import ArrayLikeFloat as ArrayLikeFloat
from pyvista.core._typing_core import ArrayLikeInt as ArrayLikeInt
from pyvista.core._typing_core import CellArrayLike as CellArrayLike
from pyvista.core._typing_core import CellsLike as CellsLike
from pyvista.core._typing_core import InteractionEventType as InteractionEventType
from pyvista.core._typing_core import LineStyle as LineStyle
from pyvista.core._typing_core import MatrixLikeBool as MatrixLikeBool
from pyvista.core._typing_core import MatrixLikeFloat as MatrixLikeFloat
from pyvista.core._typing_core import MatrixLikeInt as MatrixLikeInt
from pyvista.core._typing_core import RotationLike as RotationLike
from pyvista.core._typing_core import TransformLike as TransformLike
from pyvista.core._typing_core import VectorLikeBool as VectorLikeBool
from pyvista.core._typing_core import VectorLikeFloat as VectorLikeFloat
from pyvista.core._typing_core import VectorLikeInt as VectorLikeInt
from pyvista.core._typing_core import WrappableType as WrappableType
from pyvista.core.errors import PyVistaDeprecationWarning

if TYPE_CHECKING:
    from pyvista.core.filters.data_object import MeshValidationFields as MeshValidationFields
    from pyvista.jupyter import JupyterBackendOptions as JupyterBackendOptions
    from pyvista.plotting._typing import CameraPositionOptions as CameraPositionOptions
    from pyvista.plotting._typing import Chart as Chart
    from pyvista.plotting._typing import ColorLike as ColorLike
    from pyvista.plotting._typing import PlottableType as PlottableType

__all__ = [
    'ArrayLikeBool',
    'ArrayLikeFloat',
    'ArrayLikeInt',
    'CameraPositionOptions',
    'CellArrayLike',
    'CellsLike',
    'Chart',
    'ColorLike',
    'InteractionEventType',
    'JupyterBackendOptions',
    'LineStyle',
    'MatrixLikeBool',
    'MatrixLikeFloat',
    'MatrixLikeInt',
    'MeshValidationFields',
    'PlottableType',
    'RotationLike',
    'TransformLike',
    'VectorLikeBool',
    'VectorLikeFloat',
    'VectorLikeInt',
    'WrappableType',
]

# Imported on first access, since importing these modules here is circular or loads plotting
_LAZY_ALIASES = {
    'CameraPositionOptions': 'pyvista.plotting._typing',
    'Chart': 'pyvista.plotting._typing',
    'ColorLike': 'pyvista.plotting._typing',
    'JupyterBackendOptions': 'pyvista.jupyter',
    'MeshValidationFields': 'pyvista.core.filters.data_object',
    'PlottableType': 'pyvista.plotting._typing',
}

_MOVED_FROM_CORE = frozenset(
    {
        'CellArrayLike',
        'CellsLike',
        'InteractionEventType',
        'JupyterBackendOptions',
        'LineStyle',
        'MeshValidationFields',
        'RotationLike',
        'TransformLike',
    }
)
_MOVED_FROM_PLOTTING = frozenset({'CameraPositionOptions', 'Chart', 'ColorLike'})

# Aliases forwarded from each module with a deprecation warning
_MOVED_TO_TYPING_NAMESPACE = {
    'pyvista': _MOVED_FROM_CORE | _MOVED_FROM_PLOTTING,
    'pyvista.core': _MOVED_FROM_CORE,
    'pyvista.plotting': _MOVED_FROM_PLOTTING,
}


if not TYPE_CHECKING:  # pragma: no branch

    def __getattr__(name: str) -> object:
        """Import a type alias whose module cannot be imported with this one."""
        if name not in _LAZY_ALIASES:
            msg = f'module {__name__!r} has no attribute {name!r}'
            raise AttributeError(msg)
        alias = getattr(importlib.import_module(_LAZY_ALIASES[name]), name)
        globals()[name] = alias
        return alias


def __dir__() -> list[str]:
    """List the module attributes, including the aliases not imported yet."""
    return sorted({*globals(), *__all__})


# Deprecated type aliases with no counterpart in this module: (source, attribute, advice)
_DEPRECATED_ALIASES = {
    'Number': ('pyvista.core._typing_core._deprecated', 'Number', 'use `float` instead'),
    'NumberType': ('pyvista.core._typing_core._deprecated', '_NumberT', 'use a `TypeVar` instead'),
    'NumpyArray': (
        'pyvista.core._typing_core._deprecated',
        'NumpyArray',
        'use `numpy.typing.NDArray` instead',
    ),
    'ArrayLike': (
        'pyvista.core._typing_core._deprecated',
        '_ArrayLike',
        'use `pyvista.typing.ArrayLikeFloat`, `ArrayLikeInt` or `ArrayLikeBool` instead',
    ),
    'MatrixLike': (
        'pyvista.core._typing_core._deprecated',
        '_ArrayLike2D',
        'use `pyvista.typing.MatrixLikeFloat`, `MatrixLikeInt` or `MatrixLikeBool` instead',
    ),
    'VectorLike': (
        'pyvista.core._typing_core._deprecated',
        '_ArrayLike1D',
        'use `pyvista.typing.VectorLikeFloat`, `VectorLikeInt` or `VectorLikeBool` instead',
    ),
}


def _get_deprecated_alias(module: str, name: str) -> object:
    """Return a type alias with a warning that ``module`` no longer provides it."""
    if name in _DEPRECATED_ALIASES:
        source, attribute, advice = _DEPRECATED_ALIASES[name]
        alias = getattr(importlib.import_module(source), attribute)
        msg = f'`{module}.{name}` is deprecated; {advice}.'
    else:
        alias = getattr(sys.modules[__name__], name)
        msg = (
            f'`{module}.{name}` has moved to `pyvista.typing`; '
            f'use `pyvista.typing.{name}` instead.'
        )
    warn_external(msg, PyVistaDeprecationWarning)
    if _is_deprecation_due((0, 53)):  # pragma: no cover
        msg = 'Convert this deprecation warning into an error.'
        raise RuntimeError(msg)
    if _is_deprecation_due((0, 54)):  # pragma: no cover
        msg = f'Remove the type alias forwards from `{module}`.'
        raise RuntimeError(msg)
    return alias
