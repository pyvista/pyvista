"""Pluggable reader registry for custom file formats."""

from __future__ import annotations

import atexit
from importlib.metadata import entry_points
import pathlib
import shutil
import tempfile
from typing import TYPE_CHECKING

import pooch

from pyvista._warn_external import warn_external

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyvista.core.dataset import DataSet

_custom_ext_readers: dict[str, Callable[..., DataSet]] = {}
_custom_ext_cloud: dict[str, bool] = {}
_custom_scheme_readers: dict[str, Callable[..., DataSet]] = {}
_entry_points_loaded: bool = False
_temp_files: list[str] = []


def _cleanup_temp_files() -> None:
    """Remove temporary files created by :func:`_download_uri`."""
    for path in _temp_files:
        pathlib.Path(path).unlink(missing_ok=True)
    _temp_files.clear()


atexit.register(_cleanup_temp_files)


def _save_registry_state() -> dict:
    """Snapshot the current registry state for later restoration."""
    return {
        'ext': _custom_ext_readers.copy(),
        'cloud': _custom_ext_cloud.copy(),
        'scheme': _custom_scheme_readers.copy(),
    }


def _restore_registry_state(state: dict) -> None:
    """Restore registry state from a snapshot."""
    _custom_ext_readers.clear()
    _custom_ext_readers.update(state['ext'])
    _custom_ext_cloud.clear()
    _custom_ext_cloud.update(state['cloud'])
    _custom_scheme_readers.clear()
    _custom_scheme_readers.update(state['scheme'])


def _has_scheme(value: str) -> bool:
    """Return True if *value* starts with a URI scheme (e.g. ``https://``)."""
    # Check that :// appears before the first / to avoid false positives
    # on paths like /data/re://fresh/mesh.vtu
    slash = value.find('/')
    colon = value.find('://')
    return colon > 0 and (slash == -1 or colon < slash)


def _download_uri(uri: str, ext: str) -> str:
    """Download a remote URI to a temporary file, preserving *ext*.

    Uses ``fsspec`` when available (supports ``s3://``, ``gs://``,
    ``az://``, ``http://``, and any other registered filesystem).
    Falls back to ``pooch`` for ``http://`` and ``https://`` URIs.

    Parameters
    ----------
    uri : str
        The remote URI to download.
    ext : str
        File extension to use for the temp file (e.g. ``'.vtu'``).

    Returns
    -------
    str
        Path to the downloaded temporary file.

    Raises
    ------
    ImportError
        If the URI scheme requires ``fsspec`` and it is not installed.
    ConnectionError
        If the download fails.

    """
    suffix = ext or ''
    try:
        import fsspec  # noqa: PLC0415  — optional dependency
    except ImportError:
        if not uri.lower().startswith(('http://', 'https://')):
            scheme = uri.split('://', maxsplit=1)[0]
            msg = (
                f'Cannot download "{scheme}://" URIs without fsspec. '
                f'Install it with: pip install fsspec'
            )
            raise ImportError(msg)
        result = pooch.retrieve(uri, known_hash=None, fname=f'pyvista_download{suffix}')
        _temp_files.append(result)
        return result

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_name = tmp.name
        _temp_files.append(tmp_name)
        with fsspec.open(uri, 'rb') as remote, pathlib.Path(tmp_name).open('wb') as local:
            shutil.copyfileobj(remote, local)
    except Exception as e:
        msg = f'Failed to download "{uri}": {e}'
        raise ConnectionError(msg) from e
    else:
        return tmp_name


def _ext_supports_cloud(ext: str) -> bool:
    """Return True if the handler for *ext* was registered with ``cloud=True``."""
    _ensure_entry_points()
    return _custom_ext_cloud.get(ext, False)


def register_reader(
    key: str,
    handler: Callable[..., DataSet] | None = None,
    *,
    cloud: bool = False,
    override: bool = False,
) -> Callable[..., DataSet] | None:
    """Register a custom reader handler for a file extension or URI scheme.

    Can be used as a plain call or as a decorator.

    .. versionadded:: 0.48.0

    Parameters
    ----------
    key : str
        Either a file extension (e.g. ``'.myformat'``) or a URI scheme
        prefix (e.g. ``'s3://'``).
    handler : callable, optional
        A callable with signature ``handler(path: str, **kwargs)`` that
        returns a :class:`pyvista.DataSet`.  When omitted the function
        acts as a decorator and returns the decorated callable unchanged.
    cloud : bool, default: False
        If ``True``, the handler accepts remote URIs (``s3://``,
        ``https://``, etc.) directly.  If ``False``, PyVista will
        download HTTP(S) URIs to a temporary file before calling the
        handler.
    override : bool, default: False
        If ``True``, allow overriding a built-in VTK reader for this
        extension.

    Returns
    -------
    callable or None
        When used as a decorator (``handler`` omitted), returns the
        decorated function.  Otherwise returns ``None``.

    Raises
    ------
    ValueError
        If ``key`` collides with a built-in VTK reader and *override*
        is ``False``.

    Examples
    --------
    Register a reader for a custom file extension.

    >>> import pyvista as pv
    >>> def my_reader(path, **kwargs): ...
    >>> pv.register_reader('.myformat', my_reader)  # doctest: +SKIP

    Use as a decorator.

    >>> @pv.register_reader('.myformat')  # doctest: +SKIP
    ... def my_reader(path, **kwargs): ...

    Register a cloud-capable reader.

    >>> @pv.register_reader('.zarr', cloud=True)  # doctest: +SKIP
    ... def zarr_reader(
    ...     path, **kwargs
    ... ): ...  # can handle s3://, https://, etc. natively

    """
    if handler is None:
        # Decorator form: @pv.register_reader('.ext')
        def _decorator(fn: Callable[..., DataSet]) -> Callable[..., DataSet]:
            _register(key, fn, cloud=cloud, override=override)
            return fn

        return _decorator

    _register(key, handler, cloud=cloud, override=override)
    return None


def _register(
    key: str,
    handler: Callable[..., DataSet],
    *,
    cloud: bool = False,
    override: bool = False,
) -> None:
    """Register a handler in the appropriate registry."""
    if _has_scheme(key):
        _custom_scheme_readers[key.lower()] = handler
    else:
        key = key.lower()
        if not key.startswith('.'):
            key = f'.{key}'
        if not override:
            # Circular import: reader_registry -> reader -> fileio -> reader_registry
            from pyvista.core.utilities.reader import CLASS_READERS  # noqa: PLC0415

            if key in CLASS_READERS:
                msg = (
                    f'Cannot register custom reader for "{key}": '
                    f'collides with built-in VTK reader. '
                    f'Use override=True to replace it.'
                )
                raise ValueError(msg)
        _custom_ext_readers[key] = handler
        _custom_ext_cloud[key] = cloud


def _get_scheme_handler(filename: str) -> tuple[Callable[..., DataSet], str] | None:
    """Return ``(handler, scheme_key)`` if *filename* matches a registered URI scheme."""
    _ensure_entry_points()
    filename_lower = filename.lower()
    for scheme, handler in _custom_scheme_readers.items():
        if filename_lower.startswith(scheme):
            return handler, scheme
    return None


def _get_ext_handler(ext: str) -> Callable[..., DataSet] | None:
    """Look up a custom extension handler, discovering entry points lazily."""
    handler = _custom_ext_readers.get(ext)
    if handler is not None:
        return handler
    _ensure_entry_points()
    return _custom_ext_readers.get(ext)


def _ensure_entry_points() -> None:
    """Scan the ``pyvista.readers`` entry-point group and load handlers."""
    global _entry_points_loaded  # noqa: PLW0603
    if _entry_points_loaded:
        return
    _entry_points_loaded = True
    eps = entry_points(group='pyvista.readers')

    # Group entry points by key to detect conflicts
    seen: dict[str, list[str]] = {}
    for ep in eps:
        key = ep.name.lower()
        if not _has_scheme(key) and not key.startswith('.'):
            key = f'.{key}'
        seen.setdefault(key, []).append(ep.value)

    for ep in eps:
        key = ep.name.lower()
        if _has_scheme(key):
            registry = _custom_scheme_readers
        else:
            if not key.startswith('.'):
                key = f'.{key}'
            registry = _custom_ext_readers

        if key in registry:
            continue
        try:
            registry[key] = ep.load()
        except Exception:  # noqa: BLE001, S112
            continue

        # Warn if multiple entry points claim the same key
        if len(seen.get(key, [])) > 1:
            providers = ', '.join(seen[key])
            msg = (
                f'Multiple pyvista.readers entry points registered for '
                f'"{key}": {providers}. Using {ep.value}.'
            )
            warn_external(msg)
