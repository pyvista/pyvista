"""Single entry point for accessing any example dataset."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import difflib
from pathlib import Path
import sys
from typing import TYPE_CHECKING
from typing import Any

import pyvista as pv
from pyvista.examples._dataset_loader import _DOWNLOADABLE_TYPES
from pyvista.examples._dataset_loader import _DatasetLoader

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from pyvista.examples._dataset_loader import DatasetObject


@dataclass(frozen=True)
class Example:
    """A single example dataset: its files, where they come from, and how to read it.

    Returned by :func:`~pyvista.examples.get_example`. Every sequence-valued field is
    a tuple with one entry per path, in the same order, including for single-file
    examples.

    Notes
    -----
    The fields are limited to what an example cannot be asked for directly. Anything
    derivable is left out: the extensions are the suffixes of :attr:`paths`, the total
    size is ``sum(file_sizes)``, and the reader types and the file which is read both
    come from :attr:`readers`.

    Examples
    --------
    Look up an example. This resolves its files but does not read them.

    >>> from pyvista import examples
    >>> frog = examples.get_example('frog')
    >>> frog.name
    'frog'

    It is stored as two files, but only one of them is read.

    >>> len(frog.paths)
    2
    >>> [type(reader).__name__ for reader in frog.readers]
    ['MetaImageReader']

    Sizes are in bytes, one per path, so examples compare directly.

    >>> bunny = examples.get_example('bunny')
    >>> sum(frog.file_sizes) > sum(bunny.file_sizes)
    True

    Read the dataset itself.

    >>> mesh = frog.load()
    >>> mesh.n_cells
    31594185

    """

    name: str
    """Name of the example, such as ``'frog'``."""

    function: Callable[..., Any]
    """Public function which returns this example, such as ``examples.download_frog``."""

    paths: tuple[str, ...] = ()
    """Local path of every file or folder belonging to the example, in declaration order."""

    file_sizes: tuple[int, ...] = ()
    """Size in bytes of each entry in :attr:`paths`, one per path, folders counted in full."""

    source_urls: tuple[str, ...] = ()
    """URL each entry in :attr:`paths` is downloaded from, empty if it has none."""

    _loader: _DatasetLoader | None = field(default=None, repr=False, compare=False)

    @property
    def readers(self) -> tuple[pv.BaseReader[Any], ...]:
        """Return a reader for each file which has one.

        Empty for examples read with a custom function or generated in memory, and
        shorter than :attr:`paths` when only some files are read directly.

        Returns
        -------
        tuple[pyvista.BaseReader, ...]
            One reader per file which has one.

        """
        return tuple(r for r in getattr(self._loader, '_reader', ()) if r is not None)

    def load(self) -> DatasetObject:
        """Read the example and return its dataset.

        Returns
        -------
        DataSet | tuple[str, ...] | tuple[pyvista.BaseReader, ...]
            The dataset, as the example's own function returns it. Examples which load
            as a :class:`~pyvista.MultiBlock`, :class:`~pyvista.Texture` or
            :class:`numpy.ndarray` return that in place of a :class:`~pyvista.DataSet`.

        """
        if self._loader is None:  # pragma: no cover
            msg = f'Example {self.name!r} has no loader.'
            raise RuntimeError(msg)
        return self._loader.load()


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
    if download and isinstance(loader, _DOWNLOADABLE_TYPES):
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


def get_example(name: str | Callable[..., Any], *, download: bool = True) -> Example:
    """Look up any example dataset.

    This is a single entry point for every example in
    :mod:`pyvista.examples.examples`, :mod:`pyvista.examples.downloads`, and
    :mod:`pyvista.examples.planets`. It returns the example itself -- its files,
    where they come from, and the readers for them -- rather than the dataset, which
    :meth:`Example.load` reads. Reach for it to work with an example by name, or to
    get at its files or readers; :func:`~pyvista.examples.downloads.download_bunny`
    and its 200-odd siblings remain the direct way to load one dataset you can name
    in your source.

    Parameters
    ----------
    name : str | Callable
        Name of the example, such as ``'bunny'``, or the function which returns it,
        such as ``examples.download_bunny``. A ``'download_'`` or ``'load_'``
        prefix on the name is optional.

    download : bool, default: True
        Download the example's files if they are not already present. If ``False``,
        a ``FileNotFoundError`` is raised for any example whose files are missing.
        Files which are already cached, and examples generated in memory, are
        unaffected.

    Returns
    -------
    Example
        The example, its files, and its readers.

    See Also
    --------
    :ref:`dataset_gallery`
        Browse every available example.

    Examples
    --------
    Look up an example and read it.

    >>> from pyvista import examples
    >>> uniform = examples.get_example('uniform')
    >>> uniform.load().n_cells
    729

    Get its files, always as a tuple however many it has.

    >>> uniform.paths  # doctest:+SKIP
    ('.../pyvista/examples/uniform.vtk',)

    Get a reader for each file that has one, to inspect or configure before reading.

    >>> [type(reader).__name__ for reader in uniform.readers]
    ['VTKDataSetReader']

    """
    loader, dataset_name, function = _get_dataset_loader(name)
    return Example(
        name=dataset_name,
        function=function,
        paths=_resolve_paths(loader, dataset_name, download=download),
        file_sizes=getattr(loader, '_filesize_bytes', ()),
        source_urls=getattr(loader, 'source_url', ()),
        _loader=loader,
    )
