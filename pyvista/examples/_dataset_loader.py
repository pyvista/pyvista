"""Abstraction layer for downloading, reading, and loading dataset files.

The classes and methods in this module define an API for working with either
a single file or multiple files which may be downloaded and/or loaded as an
example dataset.

Many datasets have a straightforward input to output mapping:
    file -> read -> dataset

However, some file formats require multiple input files for reading (for example
separate data and header files):
    (file1, file1) -> read -> dataset

Or, a dataset may be combination of two separate datasets:
    file1 -> read -> dataset1 ┬─> ``combined_dataset``
    file2 -> read -> dataset2 ┘

In some cases, the input may be a folder instead of a file (for example, DICOM):
    folder -> read -> dataset

In addition, there may be a need to customize the reading function to read
files with specific options enabled (for example, set a time value), or perform
post-read processing to modify the dataset (for example, set active scalars).

This module aims to serve these use cases and provide a flexible way of
downloading, reading, and processing files with a generic mapping:
    file or files or folder -> fully processed datasets in any form

"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
import functools
import itertools
import os
from pathlib import Path
import posixpath
from typing import TYPE_CHECKING
from typing import Any
from typing import Protocol
from typing import cast
from typing import final

import pyvista as pv
from pyvista.core.utilities.fileio import get_ext

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyvista.core._typing_core import NumpyArray

# Annotations only (never resolved at runtime), so keep these out of the runtime
# namespace: ``pv.Texture`` would eagerly import rendering and break core-only builds.
if TYPE_CHECKING:
    DatasetObject = (
        pv.DataSet | pv.Texture | NumpyArray[Any] | pv.MultiBlock | pv.PartitionedDataSet
    )
    DatasetType = type[DatasetObject]


class _FileProps:
    """Define the file properties shared by single-file and multi-file loaders.

    Every per-file property is a tuple with one entry per file, so single-file and
    multi-file loaders have the same shape and callers never branch on the type.
    """

    @property
    @abstractmethod
    def paths(self) -> tuple[str, ...]:
        """Return the paths of all files."""

    @property
    def num_files(self) -> int:
        """Return the number of files across all paths.

        If a path is a folder, the number of files contained in the folder is returned.
        """
        return sum(
            1 if Path(p).is_file() else len(_get_all_nested_filepaths(p)) for p in self.paths
        )

    @property
    def unique_extensions(self) -> tuple[str, ...]:
        """Return the unique file extensions from all files."""
        return _get_unique_extensions(self.paths)

    @property
    @abstractmethod
    def _file_sizes(self) -> tuple[int, ...]:
        """Return the file sizes of all files in bytes."""

    @property
    def total_size(self) -> str:
        """Return the total size of all files formatted as a string."""
        return _format_file_size(sum(self._file_sizes))

    @property
    @abstractmethod
    def _readers(self) -> tuple[pv.BaseReader[Any] | None, ...]:
        """Return the base file readers used to read the files."""

    @property
    def unique_reader_types(self) -> tuple[type[pv.BaseReader[Any]], ...]:
        """Return unique reader types from all file readers."""
        return _get_unique_reader_types(self._readers)


class _Downloadable(Protocol):
    """Class whose files have a remote source they can be downloaded from.

    Having a source is not the same as needing to fetch one: built-in examples ship
    inside the package and also have a source URL.
    """

    @property
    @abstractmethod
    def source_names(self) -> tuple[str, ...]:
        """Return the name of each download relative to its base url."""

    @property
    @abstractmethod
    def base_urls(self) -> tuple[str, ...]:
        """Return the base url of each download."""

    @property
    def source_urls(self) -> tuple[str, ...]:
        """Return the source of each download.

        This is the full URL or local cached path used to download the data directly.
        """
        return self._source_urls()

    def _source_urls(self, *, web_blob: bool = False) -> tuple[str, ...]:
        """Join each base url with its source name, optionally as a blob url for a browser."""
        base_urls = list(self.base_urls)
        if web_blob:
            # Ensure urls are not based on a local cache path
            from pyvista.examples.downloads import _DEFAULT_VTK_DATA_SOURCE  # noqa: PLC0415
            from pyvista.examples.downloads import _FILE_CACHE  # noqa: PLC0415
            from pyvista.examples.downloads import SOURCE  # noqa: PLC0415

            for i, base in enumerate(base_urls):
                new_base = _DEFAULT_VTK_DATA_SOURCE if _FILE_CACHE and (base == SOURCE) else base
                if not new_base.startswith('http'):
                    msg = f'Expected a URL starting with "http", got {new_base!r}.'
                    raise ValueError(msg)
                new_base = new_base.replace('/raw/', '/blob/')
                if '/blob/' not in new_base:
                    msg = f'Expected "/blob/" in URL, got {new_base!r}.'
                    raise ValueError(msg)
                base_urls[i] = new_base

        # Use posixpath (not pathlib) since these are URLs, not filesystem paths.
        return tuple(
            posixpath.join(base, name)
            for base, name in zip(base_urls, self.source_names, strict=True)
        )

    @property
    def web_urls(self) -> tuple[str, ...]:
        """Return the web source of each download as blob instead of raw.

        This URL is useful for linking to the source webpage for
        a human to open on a browser.
        """
        return self._source_urls(web_blob=True)

    @property
    @abstractmethod
    def paths(self) -> tuple[str, ...]:
        """Return the file path of each downloaded file."""

    @abstractmethod
    def download(self) -> tuple[str, ...]:
        """Download and return the file paths."""


class _DatasetLoader:
    """Load a dataset."""

    def __init__(self, load_func: Callable[..., DatasetObject] | None) -> None:
        self._load_func = load_func
        self._dataset: DatasetObject | None = None

    @property
    @final
    def dataset(self) -> DatasetObject | None:
        """Return the loaded dataset objects."""
        return self._dataset

    def load(self, *args: Any, **kwargs: Any) -> DatasetObject:
        """Load and return the dataset."""
        # Subclasses should override this as needed
        if self._load_func is None:
            msg = 'No load function has been set.'
            raise RuntimeError(msg)
        return self._load_func(*args, **kwargs)

    @final
    def load_and_store_dataset(self) -> DatasetObject:
        """Load the dataset and store it."""
        dataset = self.load()
        self._dataset = dataset
        return dataset

    @final
    def clear_dataset(self) -> None:
        """Clear the stored dataset object from memory."""
        del self._dataset

    @property
    @final
    def dataset_iterable(self) -> tuple[DatasetObject, ...]:
        """Return a tuple of all dataset objects, including any nested objects.

        If the dataset is a MultiBlock, the MultiBlock itself is also returned as the first
        item. Any nested MultiBlocks are not included, only their datasets.

        For example, for a composite dataset:
            MultiBlock -> (MultiBlock, Block0, Block1, and so on)
        """
        dataset = self.dataset

        def _flat(obj: Any) -> list[Any]:
            """Recursively flatten a possibly nested sequence into a flat list."""
            if isinstance(obj, Sequence):
                output_list: list[Any] = []
                for item in obj:
                    (
                        output_list.extend(item)
                        if isinstance(item, Sequence)
                        else output_list.append(item)
                    )
                    if any(isinstance(item, Sequence) for item in output_list):
                        return _flat(output_list)
                return output_list
            else:
                return [obj]

        flat = _flat(dataset)
        if isinstance(dataset, (pv.MultiBlock, pv.PartitionedDataSet)):
            flat.insert(0, dataset)
        return tuple(flat)

    @property
    @final
    def unique_dataset_types(self) -> tuple[DatasetType, ...]:
        """Return unique dataset types from all datasets."""
        return _get_unique_dataset_types(self.dataset_iterable)

    @property
    @final
    def unique_cell_types(
        self,
    ) -> tuple[pv.CellType, ...]:
        """Return unique cell types from all datasets."""
        cell_types: dict[pv.CellType, None] = {}
        for data in self.dataset_iterable:
            # Get the underlying dataset for the texture
            dataset = (
                cast('pv.ImageData', pv.wrap(data.GetInput()))
                if isinstance(data, pv.Texture)
                else data
            )
            if not isinstance(dataset, pv.DataSet):
                # NumpyArray and MultiBlock have no cells of their own
                continue
            for cell_type in dataset.distinct_cell_types:
                cell_types[cell_type] = None
        return tuple(sorted(cell_types.keys()))


class _SingleFile(_FileProps):
    """Wrap a single file."""

    def __init__(self, path: str) -> None:
        """Wrap a single file at ``path``."""
        from pyvista.examples.downloads import USER_DATA_PATH  # noqa: PLC0415

        self._path = path if Path(path).is_absolute() else str(Path(USER_DATA_PATH) / path)

    @property
    def paths(self) -> tuple[str, ...]:
        """Return the path of the file."""
        return (self._path,)

    @property
    def _file_sizes(self) -> tuple[int, ...]:
        """Return the size of the file in bytes."""
        return (_get_file_or_folder_size(self._path),)

    @property
    def _readers(self) -> tuple[pv.BaseReader[Any] | None, ...]:
        """Return no reader; a bare file is not loaded on its own."""
        return (None,)


class _SingleFileDatasetLoader(_SingleFile, _DatasetLoader):
    """Wrap a single file for loading.

    Specify the read function and/or load functions for reading and processing the
    dataset. The read function is called on the file path first, then, if a load
    function is specified, the load function is called on the output from the read
    function.

    Parameters
    ----------
    path
        Path of the file to be loaded.

    read_func
        Specify the function used to read the file. Defaults to :func:`pyvista.read`.
        This can be used for customizing the reader's properties, or using another
        read function (for example, :func:`pyvista.read_texture` for textures). The function
        must have the file path as the first argument and should return a dataset.
        If default arguments are required by your desired read function, consider
        using :class:`functools.partial` to pre-set the arguments before passing it
        as an argument to the loader.

    load_func
        Specify the function used to load the file. Defaults to ``None``. This is typically
        used to specify any processing of the dataset after reading. The load function
        typically will accept a dataset as an input and return a dataset.

    """

    def __init__(
        self,
        path: str,
        read_func: Callable[[str], DatasetObject] | None = None,
        load_func: Callable[[DatasetObject], DatasetObject] | None = None,
    ) -> None:
        """Wrap a single file, reading it with ``read_func`` and loading it with ``load_func``."""
        _SingleFile.__init__(self, path)
        _DatasetLoader.__init__(self, load_func)
        self._read_func = pv.read if path and read_func is None else read_func

    @property
    def _readers(self) -> tuple[pv.BaseReader[Any] | None, ...]:
        # TODO: return the actual reader used, and not just a lookup
        #       (this will require an update to the 'read_func' API)
        try:
            return (pv.get_reader(self._path),)
        except (FileNotFoundError, ValueError):
            # Missing, or cannot be read directly (requires custom reader)
            return (None,)

    @property
    def loadable_paths(self) -> tuple[str, ...]:
        """Return the paths of all loadable files."""
        return (self._path,)

    def _read_and_load(
        self,
        path: str,
        read_func: Callable[[str], DatasetObject],
        load_func: Callable[[DatasetObject], DatasetObject] | None,
    ) -> DatasetObject:
        """Read ``path`` and optionally load the result."""
        read = read_func(path)
        return read if load_func is None else load_func(read)

    def load(self) -> Any:
        """Read and, if applicable, load the file."""
        path = self._path
        read_func = self._read_func
        load_func = self._load_func
        if read_func is None:
            msg = 'No read function has been set.'
            raise RuntimeError(msg)
        try:
            # Read and load normally
            return self._read_and_load(path, read_func, load_func)
        except OSError:
            # Handle error generated by pv.read if reading a directory
            if read_func is pv.read and Path(path).is_dir():
                # Re-define read function to read all files in a directory as a multiblock
                def read_dir(path: str) -> pv.MultiBlock:
                    """Read all files in ``path`` as a single MultiBlock."""
                    return _load_as_multiblock(
                        [
                            _SingleFileDatasetLoader(str(Path(path, fname.name)))
                            for fname in sorted(Path(path).iterdir())
                        ],
                    )

                return self._read_and_load(path, read_dir, load_func)
            msg = f'Error loading dataset from path:\n\t{self._path}'
            raise RuntimeError(msg)


class _DownloadableFile(_SingleFile, _Downloadable):
    """Wrap a single file which must be downloaded.

    If downloading a file from an archive, set the file path of the zip as
    ``path`` and set ``target_file`` as the file to extract. If the path is
    a zip file and no target file is specified, the entire archive is downloaded
    and extracted and the root directory of the path is returned.

    """

    def __init__(
        self,
        path: str,
        *,
        target_file: str | None = None,
        base_url: str | None = None,
        download_func: Callable[[str], str | list[str]] | None = None,
    ) -> None:
        """Wrap a single file which must be downloaded from ``base_url``."""
        _SingleFile.__init__(self, path)

        from pyvista.examples.downloads import SOURCE  # noqa: PLC0415
        from pyvista.examples.downloads import USER_DATA_PATH  # noqa: PLC0415
        from pyvista.examples.downloads import _download_archive_file_or_folder  # noqa: PLC0415
        from pyvista.examples.downloads import download_file  # noqa: PLC0415
        from pyvista.examples.downloads import file_from_files  # noqa: PLC0415
        from pyvista.examples.examples import dir_path  # noqa: PLC0415

        self._download_func: Callable[[str], str | list[str]]
        if Path(path).is_absolute():
            # Absolute path must point to a built-in dataset
            if Path(path).parent != Path(dir_path):
                msg = 'Absolute path must point to a built-in dataset.'
                raise ValueError(msg)
            # A built-in file ships inside the package. It still has a source url, so it
            # is `_Downloadable`, but downloading it only hands back the packaged path.
            self._base_url = 'https://github.com/pyvista/pyvista/raw/main/pyvista/examples/'
            self._source_name = Path(path).name
            self._download_func = self._packaged_path
        else:
            # Relative path, use vars from downloads.py
            self._base_url = base_url or SOURCE
            self._download_func = download_func or download_file
            self._source_name = path

        target_file = '' if target_file is None and (get_ext(path) == '.zip') else target_file
        if target_file is not None:
            # download from archive
            self._download_func = functools.partial(
                _download_archive_file_or_folder,
                target_file=target_file,
            )
            # The file path currently points to the archive, not the target file itself
            # Try to resolve the full path to the target file (without downloading) if
            # the archive already exists in the cache
            fullpath = None
            if Path(self._path).is_file():
                unzip_dir = Path(USER_DATA_PATH, path + '.unzip')
                extracted_files = (
                    [str(p) for p in unzip_dir.rglob('*') if p.is_file()]
                    if unzip_dir.is_dir()
                    else []
                )
                try:
                    # Get file path
                    fullpath = file_from_files(target_file, extracted_files)
                except (FileNotFoundError, RuntimeError):
                    # Get folder path
                    folder = unzip_dir / target_file
                    fullpath = str(folder) if folder.is_dir() else None
            # set the file path as the relative path of the target file if
            # the fullpath could not be resolved (i.e. not yet downloaded)
            self._path = target_file if fullpath is None else fullpath

    def _packaged_path(self, _source_name: str) -> str:
        """Return the packaged path of a built-in file, which is never downloaded."""
        return self._path

    @property
    def source_names(self) -> tuple[str, ...]:
        """Return the name of each download relative to its base url."""
        return (self._source_name,)

    @property
    def base_urls(self) -> tuple[str, ...]:
        """Return the base url of each download."""
        return (self._base_url,)

    def download(self) -> tuple[str, ...]:
        """Download the file and return its local path."""
        path = self._download_func(self._source_name)
        if isinstance(path, list):
            msg = f'Expected a single downloaded file, got multiple:\n\t{path}'
            raise TypeError(msg)
        if not (Path(path).is_file() or Path(path).is_dir()):
            msg = f'Downloaded path does not exist:\n\t{path}'
            raise RuntimeError(msg)
        # Reset the path since the full path for archive files
        # isn't known until after downloading
        self._path = path
        return (path,)


class _SingleFileDownloadableDatasetLoader(_SingleFileDatasetLoader, _DownloadableFile):
    """Wrap a single file which must first be downloaded and which can also be loaded.

    .. warning::

       ``download()`` should be called before accessing other attributes. Otherwise,
       calling ``load()`` or ``path`` may fail or produce unexpected results.

    """

    def __init__(  # noqa: PLR0917
        self,
        path: str,
        read_func: Callable[[str], DatasetObject] | None = None,
        load_func: Callable[[DatasetObject], DatasetObject] | None = None,
        target_file: str | None = None,
        download_func: Callable[[str], str | list[str]] | None = None,
        base_url: str | None = None,
    ) -> None:
        """Wrap a single file which must be downloaded before it can be loaded."""
        _SingleFileDatasetLoader.__init__(self, path, read_func=read_func, load_func=load_func)
        _DownloadableFile.__init__(
            self, path, target_file=target_file, download_func=download_func, base_url=base_url
        )


class _MultiFileDatasetLoader(_DatasetLoader, _FileProps):
    """Wrap multiple files for loading.

    Some use cases for loading multi-file examples include:

    1. Multiple input files, and each file is read/loaded independently
       E.g.: loading two separate datasets for the example
       See ``download_bolt_nut`` for a reference implementation.

    2. Multiple input files, but only one is read or loaded directly
       E.g.: loading a single dataset from a file format where data and metadata are
       stored in separate files, such as ``.raw`` and ``.mhd``.
       See ``download_head`` for a reference implementation.

    3. Multiple input files, all of which make up part of the loaded dataset
       E.g.: loading six separate image files for cubemaps
       See ``download_sky_box_cube_map`` for a reference implementation.

    Parameters
    ----------
    files_func
        Specify the function which will return a sequence of :class:`_SingleFile`
        objects required for loading the dataset.

    load_func
        Specify the function used to load the files. By default, :meth:`load()` is called
        on all the files (if loadable) and a tuple containing the loaded datasets is returned.

    """

    def __init__(
        self,
        files_func: Callable[[], Sequence[_MultiBlockFile]],
        load_func: Callable[..., DatasetObject] | None = None,
    ) -> None:
        """Wrap multiple files, produced by ``files_func``, for loading."""
        self._files_func = files_func
        self._file_loaders_: Sequence[_MultiBlockFile] | None = None
        if load_func is None:
            load_func = _load_as_dataset_or_multiblock

        _DatasetLoader.__init__(self, load_func)

    @property
    def _file_objects(self) -> Sequence[_MultiBlockFile]:
        """Return the file loaders, calling ``files_func`` once and caching the result."""
        if self._file_loaders_ is None:
            self._file_loaders_ = self._files_func()
        return self._file_loaders_

    @property
    def paths(self) -> tuple[str, ...]:
        """Return the paths of all files."""
        return tuple(itertools.chain.from_iterable(file.paths for file in self._file_objects))

    @property
    def loadable_paths(self) -> tuple[str, ...]:
        """Return the paths of all loadable files.

        A file is loadable on its own when it is wrapped as a dataset loader. When none
        of them are, the load function reads them together -- a cubemap's six faces,
        say -- so every path is returned rather than none of them.
        """
        loadable = tuple(
            itertools.chain.from_iterable(
                file.paths
                for file in self._file_objects
                if isinstance(file, _SingleFileDatasetLoader)
            ),
        )
        return loadable or self.paths

    @property
    def _file_sizes(self) -> tuple[int, ...]:
        """Return the file sizes of all files in bytes."""
        return tuple(
            itertools.chain.from_iterable(file._file_sizes for file in self._file_objects)
        )

    @property
    def _readers(self) -> tuple[pv.BaseReader[Any] | None, ...]:
        """Return the base file readers used to read the files."""
        # TODO: return the actual reader used, and not just a lookup
        #       (this will require an update to the 'read_func' API)
        return tuple(itertools.chain.from_iterable(file._readers for file in self._file_objects))

    def load(self) -> Any:
        """Load the files using the configured load function."""
        if self._load_func is None:
            msg = 'No load function has been set.'
            raise RuntimeError(msg)
        return self._load_func(self._file_objects)


class _MultiFileDownloadableDatasetLoader(
    _MultiFileDatasetLoader,
    _Downloadable,
):
    """Wrap multiple files for downloading and loading."""

    @property
    def source_names(self) -> tuple[str, ...]:
        """Return the name of each download relative to its base url."""
        return tuple(
            itertools.chain.from_iterable(
                file.source_names
                for file in self._file_objects
                if isinstance(file, _DOWNLOADABLE_TYPES)
            ),
        )

    @property
    def base_urls(self) -> tuple[str, ...]:
        """Return the base url of each download."""
        return tuple(
            itertools.chain.from_iterable(
                file.base_urls
                for file in self._file_objects
                if isinstance(file, _DOWNLOADABLE_TYPES)
            ),
        )

    def download(self) -> tuple[str, ...]:
        """Download all files and return their paths."""
        paths = tuple(
            itertools.chain.from_iterable(
                file.download()
                for file in self._file_objects
                if isinstance(file, _DOWNLOADABLE_TYPES)
            ),
        )
        if not all(Path(p).is_file() or Path(p).is_dir() for p in paths):
            msg = f'Downloaded path(s) do not exist:\n\t{paths}'
            raise RuntimeError(msg)
        return paths


def _download_dataset(
    dataset_loader: _SingleFileDownloadableDatasetLoader | _MultiFileDownloadableDatasetLoader,
    *,
    load: bool = True,
    metafiles: bool = False,
) -> Any:
    """Download and load a dataset file or files.

    Parameters
    ----------
    dataset_loader
        SingleFile or MultiFile objects of the datasets to download or load.

    load
        Read and load the file after downloading. When ``False``,
        return the path or paths to the example's files.

    metafiles
        When ``load`` is ``False``, set this value to ``True`` to
        return all files required to load the example, including any metafiles.
        If ``False``, only the paths of files which are loaded on their own are
        returned -- for a format which splits header and data into two files,
        that is the header file's path alone. When no file is loadable on its
        own, every path is returned either way.

    Returns
    -------
    Any
        Loaded dataset or paths to the example's files depending on the ``load``
        parameter. Dataset may be a texture, mesh, multiblock, array, tuple of meshes,
        or any other output loaded by the example.

    """
    # Download all files for the dataset, include any metafiles
    path = dataset_loader.download()

    # Exclude non-loadable metafiles from result (if any)
    if not metafiles and isinstance(dataset_loader, _MultiFileDownloadableDatasetLoader):
        path = dataset_loader.loadable_paths

    if load:
        return dataset_loader.load()
    # Paths are tuples internally; the public `load=False` API returns a bare path
    # whenever there is exactly one
    return path[0] if len(path) == 1 else path


_MultiBlockFile = _SingleFileDatasetLoader | _MultiFileDatasetLoader | _DownloadableFile

# For `isinstance` checks: `_Downloadable` is deliberately not `runtime_checkable`,
# since checking a runtime-checkable Protocol on Python < 3.12 evaluates properties
# such as `web_urls` and can raise.
_DOWNLOADABLE_TYPES = (_DownloadableFile, _MultiFileDownloadableDatasetLoader)


def _default_multiblock_name(file: _MultiBlockFile) -> str:
    """Return the default MultiBlock key for a loadable file, derived from its path."""
    loadable_paths: tuple[str, ...] = getattr(file, 'loadable_paths', ())
    if len(loadable_paths) != 1:
        # No single loadable path to name the block after
        return ''
    path = Path(loadable_paths[0])
    return path.name[: -len(get_ext(path.name))] if path.is_file() else path.name


def _load_as_multiblock(
    files: Sequence[_MultiBlockFile],
    names: Sequence[str] | None = None,
) -> pv.MultiBlock:
    """Load multiple files as a MultiBlock.

    This function can be used as a loading function for :class:`MultiFileLoadable`
    If the use of the ``names`` parameter is needed, use :class:`functools.partial`
    to partially specify the names parameter before passing it as loading function.
    """
    multi = pv.MultiBlock()
    if names is None:
        # set names, use filename without ext by default or dirname; one name per file
        # (not per loadable path) so this list stays aligned with `files` for `zip` below
        names = [_default_multiblock_name(file) for file in files]

    for file, name in zip(files, names, strict=True):
        # Non-loadable metafiles (e.g. a header file only used by another entry) are skipped
        if not isinstance(file, _DatasetLoader):
            continue
        loaded = file.load()
        if not isinstance(loaded, (pv.MultiBlock, pv.DataSet)):
            msg = (
                f'Only MultiBlock or DataSet objects can be loaded as a MultiBlock. '
                f'Got {type(loaded)}.'
            )
            raise TypeError(msg)
        multi.append(loaded, name)
    return multi


def _load_as_cubemap(files: str | _SingleFile | Sequence[_SingleFile]) -> pv.Texture:
    """Load multiple files as a cubemap.

    Input may be a single directory with 6 cubemap files, or a sequence
    of 6 files
    """
    if isinstance(files, str):
        return pv.cubemap(files) if Path(files).is_dir() else pv.cubemap_from_filenames(files)
    paths = list(files.paths) if isinstance(files, _SingleFile) else [f._path for f in files]
    return pv.cubemap_from_filenames(paths)


def _load_as_dataset_or_multiblock(files: Sequence[_MultiBlockFile]) -> DatasetObject:
    """Load multiple files as a MultiBlock, or as a single dataset if there is only one."""
    multiblock = _load_as_multiblock(files)
    if len(multiblock) == 1 and multiblock[0] is not None:
        return multiblock[0]
    return multiblock


def _load_and_merge(files: Sequence[_SingleFile]) -> DatasetObject:
    """Load all loadable files as separate datasets and merge them."""
    loaded = [file.load() for file in files if isinstance(file, _DatasetLoader)]
    if len(loaded) == 0:
        msg = 'No loadable files were found to merge.'
        raise ValueError(msg)
    return pv.merge(loaded)


def _get_file_or_folder_size(filepath: str) -> int:
    """Return the size of a file or folder in bytes."""
    if Path(filepath).is_file():
        return Path(filepath).stat().st_size
    if not Path(filepath).is_dir():
        msg = 'Expected a file or folder path.'
        raise ValueError(msg)
    all_filepaths = _get_all_nested_filepaths(filepath)
    return sum(Path(file).stat().st_size for file in all_filepaths)


def _format_file_size(size: int) -> str:
    """Format a size in bytes as a human-readable string (for example, ``'1.2 MB'``)."""
    size_flt = float(size)
    for unit in ('B', 'KB', 'MB'):
        if round(size_flt * 10) / 10 < 1000.0:
            return f'{int(size_flt)} {unit}' if unit == 'B' else f'{size_flt:3.1f} {unit}'
        size_flt /= 1000.0
    return f'{size_flt:.1f} GB'


def _get_file_or_folder_ext(path: str) -> str | list[str]:
    """Wrap the ``get_ext`` function to handle special cases for directories."""
    if Path(path).is_file():
        return get_ext(path)
    if not Path(path).is_dir():
        msg = 'Expected a file or folder path.'
        raise ValueError(msg)
    all_paths = _get_all_nested_filepaths(path)
    ext = [get_ext(file) for file in all_paths]
    if len(ext) == 0:
        msg = f'No files with extensions were found in"\n\t{path}'
        raise ValueError(msg)
    return ext


def _get_all_nested_filepaths(filepath: str, *, exclude_readme: bool = True) -> list[str]:
    """Walk through directory and get all file paths.

    Optionally exclude any readme files (if any).
    """
    if not (Path(filepath).is_file() or Path(filepath).is_dir()):
        msg = 'Expected a file or folder path.'
        raise ValueError(msg)

    def keep(name: str) -> bool:
        """Return whether the file ``name`` should be kept."""
        return True if not exclude_readme else not name.lower().startswith('readme')

    return next(
        [str(Path(path, name)) for name in files if keep(name)]
        for path, _, files in os.walk(filepath)
    )


def _get_unique_extensions(path: Sequence[str]) -> tuple[str, ...]:
    """Return the unique set of file extensions from a sequence of paths."""
    ext_set: set[str] = set()
    for file in path:
        ext = _get_file_or_folder_ext(file)
        ext_set.add(ext) if isinstance(ext, str) else ext_set.update(ext)
    return tuple(sorted(ext_set))


def _get_unique_reader_types(
    reader: Sequence[pv.BaseReader[Any] | None],
) -> tuple[type[pv.BaseReader[Any]], ...]:
    """Return the unique reader types from a sequence of readers, empty if there are none."""
    # Use a dict (not a set) to dedupe while keeping a deterministic, sorted order
    unique_types = dict.fromkeys(type(r) for r in reader if r is not None)
    return tuple(sorted(unique_types, key=lambda t: t.__name__))


def _get_unique_dataset_types(
    dataset_iterable: tuple[DatasetObject, ...],
) -> tuple[DatasetType, ...]:
    """Return the unique dataset types from all datasets."""
    dataset_types: dict[DatasetType, None] = {}  # use dict as an ordered set
    for dataset in dataset_iterable:
        dataset_types[type(dataset)] = None
    return tuple(dataset_types.keys())
