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
from pyvista.examples._dataset_loader import _as_str_list
from pyvista.examples._dataset_loader import _DatasetLoader
from pyvista.examples._dataset_loader import _Downloadable

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from pyvista.examples._dataset_loader import DatasetObject

_Output = Literal['dataset', 'paths', 'reader', 'metadata']
_OUTPUTS: tuple[_Output, ...] = ('dataset', 'paths', 'reader', 'metadata')


@dataclass(frozen=True)
class ExampleMetadata:
    """Metadata describing a single example dataset.

    Returned by :func:`~pyvista.examples.get_example` with ``output='metadata'``.
    Every sequence-valued field is a tuple, including for single-file examples.

    Notes
    -----
    The fields describe the example's files and where they come from. Properties of
    the dataset itself, such as its type or its cell types, require reading the files
    and are available from the dataset returned with ``output='dataset'``.

    Examples
    --------
    Get the metadata for the ``'frog'`` example.

    >>> from pyvista import examples
    >>> metadata = examples.get_example('frog', output='metadata')
    >>> metadata.name
    'frog'
    >>> metadata.num_files
    2
    >>> metadata.extensions
    ('.mhd', '.zraw')

    Only one of its two files is read to produce the dataset.

    >>> len(metadata.loadable_paths)
    1

    """

    name: str
    """Name of the example, e.g. ``'frog'``."""

    function: Callable[..., Any]
    """Public function which returns this example, e.g. ``examples.download_frog``."""

    paths: tuple[Path, ...] = ()
    """Local path of every file or folder belonging to the example, in declaration order."""

    loadable_paths: tuple[Path, ...] = ()
    """Local path of the file or files which are read to produce the dataset."""

    num_files: int = 0
    """Number of files, counting the contents of any folder in :attr:`paths`."""

    extensions: tuple[str, ...] = ()
    """Unique file extensions of the example's files."""

    file_sizes: tuple[int, ...] = ()
    """Size in bytes of each entry in :attr:`paths`, folders counted in full."""

    total_size: str = '0.0 B'
    """Total size of all files, formatted for display."""

    reader_types: tuple[type[pv.BaseReader[Any]], ...] = ()
    """Unique reader types used to read the example's files."""

    source_urls: tuple[str, ...] = ()
    """URL each entry in :attr:`paths` is downloaded from, empty if it has none."""


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


def _as_tuple(value: Any) -> tuple[Any, ...]:
    """Normalize a scalar, ``None``, or sequence into a tuple, dropping ``None`` items."""
    if value is None:
        return ()
    values = value if isinstance(value, tuple) else (value,)
    return tuple(item for item in values if item is not None)


def _resolve_paths(
    loader: _DatasetLoader, name: str, *, download: bool
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Return the example's file paths and its loadable file paths."""
    if not hasattr(loader, 'path'):
        return (), ()
    if download and isinstance(loader, _Downloadable):
        loader.download()

    # Re-read `path` after downloading: archive members only resolve once extracted
    paths = tuple(Path(p) for p in _as_str_list(loader.path))
    if missing := [str(p) for p in paths if not p.exists()]:
        missing_str = '\n\t'.join(missing)
        msg = (
            f'Example {name!r} is not available locally and download=False.\n'
            f'Missing:\n\t{missing_str}'
        )
        raise FileNotFoundError(msg)
    loadable = tuple(Path(p) for p in _as_str_list(getattr(loader, 'path_loadable', [])))
    return paths, loadable


def _collect_metadata(
    loader: _DatasetLoader,
    name: str,
    function: Callable[..., Any],
    *,
    download: bool,
) -> ExampleMetadata:
    """Gather every file and source property the loader exposes."""
    paths, loadable = _resolve_paths(loader, name, download=download)
    return ExampleMetadata(
        name=name,
        function=function,
        paths=paths,
        loadable_paths=loadable,
        num_files=getattr(loader, 'num_files', 0),
        extensions=_as_tuple(getattr(loader, 'unique_extension', None)),
        file_sizes=_as_tuple(getattr(loader, '_filesize_bytes', None)),
        total_size=getattr(loader, 'total_size', '0.0 B'),
        reader_types=_as_tuple(getattr(loader, 'unique_reader_type', None)),
        source_urls=_as_tuple(getattr(loader, 'source_url', None)),
    )


def _get_reader(loader: _DatasetLoader, name: str) -> pv.BaseReader[Any]:
    """Return the reader for the example's main file."""
    readers = _as_tuple(getattr(loader, '_reader', None))
    if not readers:
        msg = (
            f'Example {name!r} has no reader. It is either generated in memory or '
            'read with a custom function rather than a pyvista reader.'
        )
        raise ValueError(msg)
    return readers[0]


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
) -> tuple[Path, ...]: ...
@overload
def get_example(
    name: str | Callable[..., Any],
    *,
    output: Literal['reader'],
    download: bool = ...,
) -> pv.BaseReader[Any]: ...
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
    output: _Output = 'dataset',
    download: bool = True,
) -> DatasetObject | tuple[Path, ...] | pv.BaseReader[Any] | ExampleMetadata:
    """Get any example dataset, its files, its reader, or its metadata.

    This is a single entry point for every example in :mod:`pyvista.examples`,
    :mod:`pyvista.examples.examples`, :mod:`pyvista.examples.downloads`, and
    :mod:`pyvista.examples.planets`. ``get_example('bunny')`` is equivalent to
    :func:`~pyvista.examples.downloads.download_bunny`, and ``output`` selects
    what is returned instead of the dataset.

    Parameters
    ----------
    name : str | Callable
        Name of the example, e.g. ``'bunny'``, or the function which returns it,
        e.g. ``examples.download_bunny``. A ``'download_'`` or ``'load_'`` prefix
        on the name is optional.

    output : 'dataset' | 'paths' | 'reader' | 'metadata', default: 'dataset'
        What to return.

        - ``'dataset'``: the loaded dataset, as the example's own function returns it.
        - ``'paths'``: the local path of every file or folder belonging to the
          example, always as a tuple, and empty for examples generated in memory.
        - ``'reader'``: a :class:`~pyvista.BaseReader` for the example's main file.
        - ``'metadata'``: an :class:`~pyvista.examples.ExampleMetadata` describing
          the example's files and their source.

    download : bool, default: True
        Download the example's files if they are not already present. If ``False``,
        a ``FileNotFoundError`` is raised for any example whose files are missing.
        Files which are already cached, and examples generated in memory, are
        unaffected.

    Returns
    -------
    DataSet | MultiBlock | Texture | ndarray | tuple[Path, ...] | BaseReader | ExampleMetadata
        The dataset, file paths, reader, or metadata, depending on ``output``.

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
    (PosixPath('.../pyvista/examples/uniform.vtk'),)

    Get a reader for the example's main file, to inspect or configure it before
    reading.

    >>> reader = examples.get_example('uniform', output='reader')
    >>> type(reader).__name__
    'VTKDataSetReader'

    """
    if output not in _OUTPUTS:
        msg = f'Invalid output {output!r}. Must be one of: {", ".join(map(repr, _OUTPUTS))}.'
        raise ValueError(msg)

    loader, dataset_name, function = _get_dataset_loader(name)

    if output == 'metadata':
        return _collect_metadata(loader, dataset_name, function, download=download)
    if output == 'reader':
        _resolve_paths(loader, dataset_name, download=download)
        return _get_reader(loader, dataset_name)
    if output == 'paths':
        return _resolve_paths(loader, dataset_name, download=download)[0]

    _resolve_paths(loader, dataset_name, download=download)
    return loader.load()
