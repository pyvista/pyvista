"""Type aliases for annotating code that uses PyVista."""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING

from pyvista.core._typing_core import ArrayLike as ArrayLike
from pyvista.core._typing_core import CellArrayLike as CellArrayLike
from pyvista.core._typing_core import CellsLike as CellsLike
from pyvista.core._typing_core import InteractionEventType as InteractionEventType
from pyvista.core._typing_core import LineStyle as LineStyle
from pyvista.core._typing_core import MatrixLike as MatrixLike
from pyvista.core._typing_core import RotationLike as RotationLike
from pyvista.core._typing_core import TransformLike as TransformLike
from pyvista.core._typing_core import VectorLike as VectorLike

if TYPE_CHECKING:
    from pyvista.core.filters.data_object import MeshValidationFields as MeshValidationFields
    from pyvista.jupyter import JupyterBackendOptions as JupyterBackendOptions
    from pyvista.plotting._typing import CameraPositionOptions as CameraPositionOptions
    from pyvista.plotting._typing import Chart as Chart
    from pyvista.plotting._typing import ColorLike as ColorLike

__all__ = [
    'ArrayLike',
    'CameraPositionOptions',
    'CellArrayLike',
    'CellsLike',
    'Chart',
    'ColorLike',
    'InteractionEventType',
    'JupyterBackendOptions',
    'LineStyle',
    'MatrixLike',
    'MeshValidationFields',
    'RotationLike',
    'TransformLike',
    'VectorLike',
]

# Imported on first access, since importing these modules here is circular or loads plotting
_LAZY_ALIASES = {
    'CameraPositionOptions': 'pyvista.plotting._typing',
    'Chart': 'pyvista.plotting._typing',
    'ColorLike': 'pyvista.plotting._typing',
    'JupyterBackendOptions': 'pyvista.jupyter',
    'MeshValidationFields': 'pyvista.core.filters.data_object',
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
_REMOVED_ALIASES = {
    'Number': ('pyvista.core._typing_core._aliases', 'Number', 'use `float` instead'),
    'NumberType': ('pyvista.core._typing_core._array_like', '_NumberT', 'use a `TypeVar` instead'),
    'NumpyArray': (
        'pyvista.core._typing_core._array_like',
        'NumpyArray',
        'use `numpy.typing.NDArray` instead',
    ),
}


def _get_deprecated_alias(module: str, name: str) -> object:
    """Return a type alias with a warning that ``module`` no longer provides it."""
    from pyvista._version import _is_deprecation_due  # noqa: PLC0415
    from pyvista._warn_external import warn_external  # noqa: PLC0415
    from pyvista.core.errors import PyVistaDeprecationWarning  # noqa: PLC0415

    if name in _REMOVED_ALIASES:
        source, attribute, advice = _REMOVED_ALIASES[name]
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
