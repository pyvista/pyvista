"""Formats served by a companion package rather than by PyVista itself."""

from __future__ import annotations

import importlib
import importlib.util
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import NamedTuple
from typing import cast

if TYPE_CHECKING:
    from collections.abc import Callable


class _OptionalFormat(NamedTuple):
    """A format whose reader, writer, or both live in a companion package."""

    module: str
    distribution: str
    extra: str
    describes: str
    read_attr: str | None = None
    write_attr: str | None = None
    reader_class: str | None = None


class _Side(NamedTuple):
    """The read or the write half of an optional format."""

    field: Literal['read_attr', 'write_attr']
    gerund: str
    verb: str
    noun: str


_READ = _Side(field='read_attr', gerund='Reading', verb='read', noun='reader')
_WRITE = _Side(field='write_attr', gerund='Writing', verb='save', noun='writer')


_ZSTD = _OptionalFormat(
    module='pyvista_zstd',
    distribution='pyvista-zstd',
    extra='pyvista[io]',
    describes="PyVista's native zstd-compressed format",
    read_attr='read',
    write_attr='write',
    reader_class='pyvista_zstd.Reader',
)

_OPTIONAL_FORMATS: dict[str, _OptionalFormat] = {
    '.frd': _OptionalFormat(
        module='pyvista_frd',
        distribution='pyvista-frd-reader',
        extra='pyvista[io]',
        describes='CalculiX FRD result files',
        read_attr='read',
        reader_class='pyvista_frd.FRDReader',
    ),
    '.pv': _ZSTD,
    # Read-only: pyvista-zstd still writes .zvtk, but warns that .pv supersedes it.
    '.zvtk': _ZSTD._replace(describes='legacy zstd-compressed files', write_attr=None),
}


def _format_for(ext: str, side: _Side) -> _OptionalFormat | None:
    """Return the optional format serving ``ext`` on ``side``, if any."""
    fmt = _OPTIONAL_FORMATS.get(ext)
    if fmt is None or getattr(fmt, side.field) is None:
        return None
    return fmt


def _installed_extensions(side: _Side) -> set[str]:
    """Return every extension optional on ``side`` whose package looks importable."""
    installed = set()
    for ext, fmt in _OPTIONAL_FORMATS.items():
        if _format_for(ext, side) is None:
            continue
        try:
            found = importlib.util.find_spec(fmt.module) is not None
        except ImportError:
            found = False
        if found:
            installed.add(ext)
    return installed


def _import_handler(ext: str, side: _Side) -> tuple[Callable[..., Any] | None, ImportError | None]:
    """Import the companion package serving ``ext``, returning its handler or the failure."""
    fmt = _format_for(ext, side)
    if fmt is None:
        return None, None
    try:
        module = importlib.import_module(fmt.module)
    except ImportError as err:
        return None, err
    return cast('Callable[..., Any]', getattr(module, getattr(fmt, side.field))), None


def _missing_message(ext: str, side: _Side, filename: str | None = None) -> str | None:
    """Return install instructions when ``ext`` needs a package PyVista cannot import."""
    fmt = _format_for(ext, side)
    if fmt is None:
        return None
    handler, err = _import_handler(ext, side)
    if handler is not None:
        return None
    opening = f'Cannot {side.verb} {filename!r}.\n' if filename is not None else ''
    installed = not (isinstance(err, ModuleNotFoundError) and err.name == fmt.module)
    if installed:
        return (
            f'{opening}{side.gerund} {fmt.describes} ({ext}) requires the '
            f'`{fmt.distribution}` package, which is installed but failed to import:\n'
            f'    {err}'
        )
    return (
        f'{opening}{side.gerund} {fmt.describes} ({ext}) requires the '
        f'`{fmt.distribution}` package, which is not installed.\n'
        f'\n'
        f'Install it with:\n'
        f'    pip install {fmt.extra}\n'
        f'\n'
        f'or, for just this {side.noun}:\n'
        f'    pip install {fmt.distribution}'
    )


def _source(ext: str, side: _Side) -> str:
    """Return the ``module:attr`` origin recorded for a resolved optional format."""
    fmt = _OPTIONAL_FORMATS[ext]
    return f'{fmt.module}:{getattr(fmt, side.field)}'
