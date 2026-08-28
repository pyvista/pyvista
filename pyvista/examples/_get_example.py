"""Single entry point for accessing any example dataset."""

from __future__ import annotations

from dataclasses import dataclass
import difflib
from pathlib import Path
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import overload

import pyvista as pv
from pyvista.examples._dataset_loader import _DatasetLoader
from pyvista.examples._dataset_loader import _Downloadable

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from pyvista.examples._dataset_loader import DatasetObject

_Output = Literal['dataset', 'paths', 'readers', 'metadata']
_OUTPUTS: tuple[_Output, ...] = ('dataset', 'paths', 'readers', 'metadata')


@dataclass(frozen=True)
class ExampleMetadata:
    """Metadata describing a single example dataset.

    Returned by :func:`~pyvista.examples.get_example` with ``output='metadata'``.
    Every sequence-valued field is a tuple, including for single-file examples.

    Notes
    -----
    The fields are limited to what an example cannot be asked for directly. Anything
    derivable is left out: the extensions are the suffixes of ``paths``, the total size
    is ``sum(file_sizes)``, and the reader types and the file which is read both come
    from the readers returned with ``output='readers'``.

    Examples
    --------
    Get the metadata for the ``'frog'`` example, which is stored as two files.

    >>> from pyvista import examples
    >>> metadata = examples.get_example('frog', output='metadata')
    >>> metadata.name
    'frog'
    >>> len(metadata.paths)
    2

    Sizes are in bytes, one per path, so examples compare directly.

    >>> bunny = examples.get_example('bunny', output='metadata')
    >>> sum(metadata.file_sizes) > sum(bunny.file_sizes)
    True

    """

    name: str
    """Name of the example, such as ``'frog'``."""

    function: Callable[..., Any]
    """Public function which returns this example, such as ``examples.download_frog``."""

    paths: tuple[str, ...] = ()
    """Local path of every file or folder belonging to the example, in declaration order."""

    file_sizes: tuple[int, ...] = ()
    """Size in bytes of each entry in ``paths``, one per path, folders counted in full."""

    source_urls: tuple[str, ...] = ()
    """URL each entry in ``paths`` is downloaded from, empty if it has none."""


def _supported_modules() -> tuple[ModuleType, ...]:
    """Return the modules which define example dataset loaders."""
    return (pv.examples.examples, pv.examples.downloads, pv.examples.planets)


def _example_loader(module: ModuleType, name: str) -> _DatasetLoader | None:
    """Return the loader ``module`` defines for an example, if it defines one."""
    loader = getattr(module, '_dataset_' + name, None)
    return loader if isinstance(loader, _DatasetLoader) else None


def _example_names(module: ModuleType) -> list[str]:
    """Return the name of every example defined by ``module``."""
    names = (
        attr.removeprefix('_dataset_') for attr in vars(module) if attr.startswith('_dataset_')
    )
    return [name for name in names if _example_loader(module, name) is not None]


def _public_function(module: ModuleType, name: str) -> Callable[..., Any]:
    """Return the public ``download_``/``load_`` function for an example name."""
    for prefix in ('download_', 'load_'):
        func = getattr(module, prefix + name, None)
        if func is not None:
            return func
    msg = f'Example {name!r} has no public function in {module.__name__!r}.'
    raise ValueError(msg)


def _get_dataset_loader(
    name: str | Callable[..., Any],
) -> tuple[_DatasetLoader, str, Callable[..., Any]]:
    """Return the loader, name, and public function for an example."""
    if callable(name):
        dataset_name = name.__name__.removeprefix('download_').removeprefix('load_')
        loader = _example_loader(sys.modules[name.__module__], dataset_name)
        if loader is None:
            msg = f'Function {name.__name__!r} does not have an example dataset.'
            raise ValueError(msg)
        return loader, dataset_name, name

    dataset_name = name.removeprefix('download_').removeprefix('load_')
    for module in _supported_modules():
        loader = _example_loader(module, dataset_name)
        if loader is not None:
            return loader, dataset_name, _public_function(module, dataset_name)

    available = sorted(
        example for module in _supported_modules() for example in _example_names(module)
    )
    msg = f'Example {dataset_name!r} does not exist.'
    if close := difflib.get_close_matches(dataset_name, available, n=3):
        msg += f' Did you mean: {", ".join(map(repr, close))}?'
    raise ValueError(msg)


def _resolve_paths(loader: _DatasetLoader, name: str, *, download: bool) -> tuple[str, ...]:
    """Return the example's file paths, downloading them first if allowed."""
    if not hasattr(loader, 'path'):
        return ()
    if download and isinstance(loader, _Downloadable):
        loader.download()

    # Re-read `path` after downloading: archive members only resolve once extracted
    paths = tuple(loader.path)
    if missing := [p for p in paths if not Path(p).exists()]:
        missing_str = '\n\t'.join(missing)
        msg = (
            f'Example {name!r} is not available locally and download=False.\n'
            f'Missing:\n\t{missing_str}'
        )
        raise FileNotFoundError(msg)
    return paths


def _collect_metadata(
    loader: _DatasetLoader,
    name: str,
    function: Callable[..., Any],
    *,
    download: bool,
) -> ExampleMetadata:
    """Gather every file and source property the loader exposes."""
    return ExampleMetadata(
        name=name,
        function=function,
        paths=_resolve_paths(loader, name, download=download),
        file_sizes=getattr(loader, '_filesize_bytes', ()),
        source_urls=getattr(loader, 'source_url', ()),
    )


def _get_readers(loader: _DatasetLoader) -> tuple[pv.BaseReader[Any], ...]:
    """Return a reader for each of the example's files which has one."""
    return tuple(r for r in getattr(loader, '_reader', ()) if r is not None)


@overload
def get_example(
    name: str | Callable[..., Any],
    *,
    output: Literal['dataset'] = ...,
    download: bool = ...,
) -> DatasetObject: ...
@overload
def get_example(
    name: str | Callable[..., Any],
    *,
    output: Literal['paths'],
    download: bool = ...,
) -> tuple[str, ...]: ...
@overload
def get_example(
    name: str | Callable[..., Any],
    *,
    output: Literal['readers'],
    download: bool = ...,
) -> tuple[pv.BaseReader[Any], ...]: ...
@overload
def get_example(
    name: str | Callable[..., Any],
    *,
    output: Literal['metadata'],
    download: bool = ...,
) -> ExampleMetadata: ...
def get_example(
    name: str | Callable[..., Any],
    *,
    output: Literal['dataset', 'paths', 'readers', 'metadata'] = 'dataset',
    download: bool = True,
) -> DatasetObject | tuple[str, ...] | tuple[pv.BaseReader[Any], ...] | ExampleMetadata:
    """Get any example dataset, its files, its reader, or its metadata.

    This is a single entry point for every example in
    :mod:`pyvista.examples.examples`, :mod:`pyvista.examples.downloads`, and
    :mod:`pyvista.examples.planets`. ``get_example('bunny')`` is equivalent to
    :func:`~pyvista.examples.downloads.download_bunny`, and ``output`` selects
    what is returned instead of the dataset.

    Parameters
    ----------
    name : str | Callable
        Name of the example, such as ``'bunny'``, or the function which returns it,
        such as ``examples.download_bunny``. A ``'download_'`` or ``'load_'``
        prefix on the name is optional.

    output : 'dataset' | 'paths' | 'readers' | 'metadata', default: 'dataset'
        What to return.

        - ``'dataset'``: the loaded dataset, as the example's own function returns it.
        - ``'paths'``: the local path of every file or folder belonging to the
          example, always as a tuple, and empty for examples generated in memory.
        - ``'readers'``: a :class:`~pyvista.BaseReader` for each file which has one,
          always as a tuple, and empty for examples read with a custom function or
          generated in memory.
        - ``'metadata'``: an :class:`~pyvista.examples.ExampleMetadata` describing
          the example's files and their source.

    download : bool, default: True
        Download the example's files if they are not already present. If ``False``,
        a ``FileNotFoundError`` is raised for any example whose files are missing.
        Files which are already cached, and examples generated in memory, are
        unaffected.

    Returns
    -------
    DataSet | tuple[str, ...] | tuple[pyvista.BaseReader, ...] | ExampleMetadata
        The dataset, its file paths, its readers, or its metadata, depending on
        ``output``. Examples which load as a :class:`~pyvista.MultiBlock`,
        :class:`~pyvista.Texture` or :class:`numpy.ndarray` return that in place of a
        :class:`~pyvista.DataSet`.

    See Also
    --------
    :ref:`dataset_gallery`
        Browse every available example.

    Examples
    --------
    Load an example by name.

    >>> from pyvista import examples
    >>> mesh = examples.get_example('uniform')
    >>> mesh.n_cells
    729

    Get the files instead. Every example returns a tuple, however many files it has.

    >>> examples.get_example('uniform', output='paths')  # doctest:+SKIP
    ('.../pyvista/examples/uniform.vtk',)

    Get a reader for each file that has one, to inspect or configure before reading.

    >>> readers = examples.get_example('uniform', output='readers')
    >>> [type(reader).__name__ for reader in readers]
    ['VTKDataSetReader']

    """
    if output not in _OUTPUTS:
        msg = f'Invalid output {output!r}. Must be one of: {", ".join(map(repr, _OUTPUTS))}.'
        raise ValueError(msg)

    loader, dataset_name, function = _get_dataset_loader(name)

    if output == 'metadata':
        return _collect_metadata(loader, dataset_name, function, download=download)
    if output == 'readers':
        _resolve_paths(loader, dataset_name, download=download)
        return _get_readers(loader)
    if output == 'paths':
        return _resolve_paths(loader, dataset_name, download=download)

    _resolve_paths(loader, dataset_name, download=download)
    return loader.load()
