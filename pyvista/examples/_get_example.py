"""Single entry point for accessing any example dataset."""

from __future__ import annotations

from dataclasses import dataclass
import difflib
from pathlib import Path
import sys
from typing import TYPE_CHECKING
from typing import Any

import pyvista as pv
from pyvista.examples._dataset_loader import _DOWNLOADABLE_TYPES
from pyvista.examples._dataset_loader import _DatasetLoader
from pyvista.examples._dataset_loader import _FileProps

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from pyvista.examples._dataset_loader import DatasetObject


@dataclass(frozen=True)
class Example:
    """A single example dataset: its files, where they come from, and how to read it.

    .. versionadded:: 0.49

    Returned by :func:`~pyvista.examples.get_example`. Every sequence-valued field is
    a tuple with one entry per path, in the same order, including for single-file
    examples.

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
    >>> len(frog.readers)
    1

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
    """Public function which returns this example's dataset, such as ``examples.download_frog``."""

    paths: tuple[str, ...] = ()
    """Local path of every file or folder belonging to the example, in declaration order."""

    file_sizes: tuple[int, ...] = ()
    """Size in bytes of each entry in ``paths``, one per path, folders counted in full."""

    source_urls: tuple[str, ...] = ()
    """URL each entry in ``paths`` is downloaded from, empty if it has none."""

    @property
    def _loader(self) -> _DatasetLoader:
        """Return the loader backing this example, resolved from :attr:`function`."""
        loader, _, _ = _get_dataset_loader(self.function)
        return loader

    @property
    def readers(self) -> tuple[pv.BaseReader[Any], ...]:
        """Return a reader for each file which has one.

        The readers report which reader PyVista resolves for each file. They are not
        the objects :meth:`load` reads through, so configuring one does not change what
        :meth:`load` returns.

        Empty for examples read with a custom function or generated in memory, and
        shorter than :attr:`paths` when only some files are read directly.

        Returns
        -------
        tuple[pyvista.BaseReader, ...]
            One reader per file which has one.

        """
        loader = self._loader
        if not isinstance(loader, _FileProps):
            return ()
        return tuple(r for r in loader._reader if r is not None)

    def load(self) -> DatasetObject:
        """Read the example and return its dataset.

        Returns
        -------
        DataSet | pyvista.MultiBlock | pyvista.Texture | numpy.ndarray
            The dataset the example's own :attr:`function` returns, read from
            :attr:`paths`.

        """
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
        module = sys.modules[name.__module__]
        loader = _example_loader(module, dataset_name)
        if loader is None:
            msg = f'Function {name.__name__!r} does not have an example dataset.'
            raise ValueError(msg)
        # Several names can share a stem, and only one of them owns the dataset:
        # `planets` has both `download_saturn_rings` and a deprecated `load_saturn_rings`
        canonical = _public_function(module, dataset_name)
        if canonical is not name:
            msg = (
                f'Function {name.__name__!r} is not the function for example '
                f'{dataset_name!r}; that is {canonical.__name__!r}.'
            )
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
    downloaded = download and isinstance(loader, _DOWNLOADABLE_TYPES)
    if downloaded:
        loader.download()
    if not isinstance(loader, _FileProps):
        return ()

    # Re-read `path` after downloading: archive members only resolve once extracted
    # A relative path means an archive member which `download()` has not resolved yet.
    # `Path` would resolve it against the working directory, and `Path('')` against `'.'`,
    # so both would look present and hand back whatever the caller happens to be sitting in.
    paths = tuple(loader.path)
    if missing := [p for p in paths if not (p and Path(p).is_absolute() and Path(p).exists())]:
        missing_str = '\n\t'.join(missing)
        if not download:
            reason = 'and download=False'
        elif downloaded:
            reason = 'even after downloading'
        else:
            reason = 'and cannot be downloaded'
        msg = f'Example {name!r} is not available locally {reason}.\nMissing:\n\t{missing_str}'
        raise FileNotFoundError(msg)
    return paths


def get_example(name: str | Callable[..., Any], *, download: bool = True) -> Example:
    """Look up any example dataset.

    .. versionadded:: 0.49

    This is a single entry point for every example in
    :mod:`pyvista.examples.examples`, :mod:`pyvista.examples.downloads`, and
    :mod:`pyvista.examples.planets`. It returns the example itself -- its files,
    where they come from, and the readers for them -- rather than the dataset, which
    :meth:`Example.load` reads. Reach for it to work with an example by name, or to
    get at its files or readers.

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
    >>> mesh = uniform.load()
    >>> mesh.n_cells
    729

    Get its files, always as a tuple however many it has.

    >>> uniform.paths  # doctest:+SKIP
    ('.../pyvista/examples/uniform.vtk',)

    Get the reader PyVista resolves for each file that has one. Most examples have
    exactly one reader.

    >>> [type(reader).__name__ for reader in uniform.readers]
    ['VTKDataSetReader']

    """
    loader, dataset_name, function = _get_dataset_loader(name)
    return Example(
        name=dataset_name,
        function=function,
        paths=_resolve_paths(loader, dataset_name, download=download),
        file_sizes=loader._filesize_bytes if isinstance(loader, _FileProps) else (),
        source_urls=loader.source_url if isinstance(loader, _DOWNLOADABLE_TYPES) else (),
    )
