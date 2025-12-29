"""Abstraction layer for downloading, reading, and loading dataset files.

The classes and methods in this module define an API for working with either
a single file or multiple files which may be downloaded and/or loaded as an
example dataset.

Many datasets have a straightforward input to output mapping:
    file -> read -> dataset

However, some file formats require multiple input files for reading (e.g.
separate data and header files):
    (file1, file1) -> read -> dataset

Or, a dataset may be combination of two separate datasets:
    file1 -> read -> dataset1 ┬─> combined_dataset
    file2 -> read -> dataset2 ┘

In some cases, the input may be a folder instead of a file (e.g. DICOM):
    folder -> read -> dataset

In addition, there may be a need to customize the reading function to read
files with specific options enabled (e.g. set a time value), or perform
post-read processing to modify the dataset (e.g. set active scalars).

This module aims to serve these use cases and provide a flexible way of
downloading, reading, and processing files with a generic mapping:
    file or files or folder -> fully processed dataset(s) in any form

"""

# mypy: disable-error-code="redundant-expr"
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
import functools
import os
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Generic
from typing import Protocol
from typing import TypeVar
from typing import cast
from typing import final
from typing import runtime_checkable

import pyvista as pv
from pyvista.core._typing_core import NumpyArray
from pyvista.core.utilities.fileio import get_ext

if TYPE_CHECKING:
    from collections.abc import Callable

# Define TypeVars for two main class definitions used by this module:
#   1. classes for single file inputs: T -> T
#   2. classes for multi-file inputs: (T, ...) -> (T, ...)
# Any properties with these typevars should have a one-to-one mapping for all files
_FilePropStrType_co = TypeVar(
    '_FilePropStrType_co',
    str,
    tuple[str, ...],
    covariant=True,
)
_FilePropIntType_co = TypeVar(
    '_FilePropIntType_co',
    int,
    tuple[int, ...],
    covariant=True,
)

DatasetObject = pv.DataSet | pv.Texture | NumpyArray[Any] | pv.MultiBlock
DatasetType = type[pv.DataSet] | type[pv.Texture] | type[NumpyArray[Any]] | type[pv.MultiBlock]


class _BaseFilePropsProtocol(Generic[_FilePropStrType_co, _FilePropIntType_co]):
    @property
    @abstractmethod
    def path(self) -> _FilePropStrType_co:
        """Return the path(s) of all files."""

    @property
    def num_files(self) -> int:
        """Return the number of files from path or paths.

        If a path is a folder, the number of files contained in the folder is returned.
        """
        path = self.path
        paths = [path] if isinstance(path, str) else path
        return sum(1 if os.path.isfile(p) else len(_get_all_nested_filepaths(p)) for p in paths)

    @property
    def unique_extension(self) -> str | tuple[str, ...]:
        """Return the unique file extension(s) from all files."""
        return _get_unique_extension(self.path)

    @property
    @abstractmethod
    def _filesize_bytes(self) -> _FilePropIntType_co:
        """Return the file size(s) of all files in bytes."""

    @property
    @abstractmethod
    def _filesize_format(self) -> _FilePropStrType_co:
        """Return the formatted size of all file(s)."""

    @property
    @abstractmethod
    def _total_size_bytes(self) -> int:
        """Return the total size of all files in bytes."""

    @property
    @abstractmethod
    def total_size(self) -> str:
        """Return the total size of all files formatted as a string."""

    @property
    @abstractmethod
    def _reader(
        self,
    ) -> pv.BaseReader | tuple[pv.BaseReader | None, ...] | None:
        """Return the base file reader(s) used to read the files."""

    @property
    def unique_reader_type(
        self,
    ) -> type[pv.BaseReader] | tuple[type[pv.BaseReader], ...] | None:
        """Return unique reader type(s) from all file readers."""
        return _get_unique_reader_type(self._reader)


class _SingleFilePropsProtocol(_BaseFilePropsProtocol[str, int]):
    """Define file properties of a single file."""


class _MultiFilePropsProtocol(
    _BaseFilePropsProtocol[tuple[str, ...], tuple[int, ...]],
):
    """Define file properties of multiple files."""


@runtime_checkable
class _Downloadable(Protocol[_FilePropStrType_co]):
    """Class which downloads file(s) from a source."""

    @property
    @abstractmethod
    def source_name(self) -> _FilePropStrType_co:
        """Return the name of the download relative to the base url."""

    @property
    @abstractmethod
    def base_url(self) -> _FilePropStrType_co:
        """Return the base url of the download."""

    @property
    def source_url_raw(self) -> _FilePropStrType_co:
        """Return the raw source of the download.

        This is the full URL used to download the data directly.
        """
        name = self.source_name
        name_iter = [name] if isinstance(name, str) else name
        url = self.base_url
        base_url_iter = [url] if isinstance(url, str) else url
        url_raw = list(map(os.path.join, base_url_iter, name_iter))
        return url_raw[0] if isinstance(name, str) else tuple(url_raw)

    @property
    def source_url_blob(self) -> _FilePropStrType_co:
        """Return the blob source of the download.

        This URL is useful for linking to the source webpage for
        a human to open on a browser.
        """
        # Make single urls iterable and replace 'raw' with 'blob'
        url_raw = self.source_url_raw
        url_iter = [url_raw] if isinstance(url_raw, str) else url_raw
        url_blob = [url.replace('/raw/', '/blob/') for url in url_iter]
        return url_blob[0] if isinstance(url_raw, str) else tuple(url_blob)

    @property
    @abstractmethod
    def path(self) -> _FilePropStrType_co:
        """Return the file path of downloaded file."""

    @abstractmethod
    def download(self) -> _FilePropStrType_co:
        """Download and return the file path(s)."""


class _DatasetLoader:
    """Load a dataset."""

    def __init__(self, load_func: Callable[..., DatasetObject]):
        self._load_func = load_func
        self._dataset: DatasetObject | None = None

    @property
    @final
    def dataset(self) -> DatasetObject | None:
        """Return the loaded dataset object(s)."""
        return self._dataset

    def load(self, *args, **kwargs) -> DatasetObject:
        """Load and return the dataset."""
        # Subclasses should override this as needed
        return self._load_func(*args, **kwargs)

    @final
    def load_and_store_dataset(self) -> DatasetObject:
        """Load the dataset and store it."""
        dataset = self.load()
        self._dataset = dataset
        return dataset

    @final
    def clear_dataset(self):
        """Clear the stored dataset object from memory."""
        del self._dataset

    @property
    @final
    def dataset_iterable(self) -> tuple[DatasetObject, ...]:
        """Return a tuple of all dataset object(s), including any nested objects.

        If the dataset is a MultiBlock, the MultiBlock itself is also returned as the first
        item. Any nested MultiBlocks are not included, only their datasets.

        E.g. for a composite dataset:
            MultiBlock -> (MultiBlock, Block0, Block1, ...)
        """
        dataset = self.dataset

        def _flat(obj):
            if isinstance(obj, Sequence):
                output_list = []  # type: ignore[var-annotated]
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
        if isinstance(dataset, pv.MultiBlock):
            flat.insert(0, dataset)
        return tuple(flat)

    @property
    @final
    def unique_dataset_type(
        self,
    ) -> DatasetType | tuple[DatasetType, ...] | None:
        """Return unique dataset type(s) from all datasets."""
        return _get_unique_dataset_type(self.dataset_iterable)

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
            try:
                if isinstance(dataset, pv.ExplicitStructuredGrid):
                    # extract_cells_by_type does not support this datatype
                    # so get cells manually
                    cells = (c.type for c in dataset.cell)
                    [cell_types.update({cell_type: None}) for cell_type in cells]
                else:
                    for cell_type in pv.CellType:
                        extracted = dataset.extract_cells_by_type(cell_type)  # type: ignore[union-attr]
                        if extracted.n_cells > 0:
                            cell_types[cell_type] = None
            except AttributeError:
                continue
        return tuple(sorted(cell_types.keys()))


class _SingleFile(_SingleFilePropsProtocol):
    """Wrap a single file."""

    def __init__(self, path):
        from pyvista.examples.downloads import USER_DATA_PATH  # noqa: PLC0415

        self._path = (
            path if path is None or os.path.isabs(path) else os.path.join(USER_DATA_PATH, path)
        )

    @property
    def path(self) -> str:
        return self._path

    @property
    def _filesize_bytes(self) -> int:
        return _get_file_or_folder_size(self.path)

    @property
    def _filesize_format(self) -> str:
        return _format_file_size(self._filesize_bytes)

    @property
    def _total_size_bytes(self) -> int:
        return self._filesize_bytes

    @property
    def total_size(self) -> str:
        return self._filesize_format

    @property
    def _reader(self) -> pv.BaseReader | None:
        return None


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
        read function (e.g. :func:`pyvista.read_texture` for textures). The function
        must have the file path as the first argument and should return a dataset.
        If default arguments are required by your desired read function, consider
        using :class:`functools.partial` to pre-set the arguments before passing it
        as an argument to the loader.

    load_func
        Specify the function used to load the file. Defaults to `None`. This is typically
        used to specify any processing of the dataset after reading. The load function
        typically will accept a dataset as an input and return a dataset.

    """

    def __init__(
        self,
        path: str,
        read_func: Callable[[str], DatasetType] | None = None,
        load_func: Callable[[DatasetType], Any] | None = None,
    ):
        _SingleFile.__init__(self, path)
        _DatasetLoader.__init__(self, load_func)  # type: ignore[arg-type]
        self._read_func = pv.read if path and read_func is None else read_func

    @property
    def _reader(self) -> pv.BaseReader | None:
        # TODO: return the actual reader used, and not just a lookup
        #       (this will require an update to the 'read_func' API)
        try:
            return pv.get_reader(self.path)
        except ValueError:
            # Cannot be read directly (requires custom reader)
            return None

    @property
    def path_loadable(self) -> str:
        return self.path

    def load(self):
        path = self.path
        read_func = self._read_func
        load_func = self._load_func
        try:
            # Read and load normally
            return read_func(path) if load_func is None else load_func(read_func(path))  # type: ignore[misc]
        except OSError:
            # Handle error generated by pv.read if reading a directory
            if read_func is pv.read and Path(path).is_dir():
                # Re-define read function to read all files in a directory as a multiblock
                read_func = lambda path: _load_as_multiblock(  # type: ignore[assignment, return-value]
                    [
                        _SingleFileDatasetLoader(str(Path(path, fname)))
                        for fname in sorted(os.listdir(path))  # noqa: PTH208
                    ],
                )
                return read_func(path) if load_func is None else load_func(read_func(path))
            else:
                msg = f'Error loading dataset from path:\n\t{self.path}'
                raise RuntimeError(msg)


class _DownloadableFile(_SingleFile, _Downloadable[str]):
    """Wrap a single file which must be downloaded.

    If downloading a file from an archive, set the filepath of the zip as
    ``path`` and set ``target_file`` as the file to extract. If the path is
    a zip file and no target file is specified, the entire archive is downloaded
    and extracted and the root directory of the path is returned.

    """

    def __init__(
        self,
        path: str,
        target_file: str | None = None,
    ):
        _SingleFile.__init__(self, path)

        from pyvista.examples.downloads import SOURCE  # noqa: PLC0415
        from pyvista.examples.downloads import USER_DATA_PATH  # noqa: PLC0415
        from pyvista.examples.downloads import _download_archive_file_or_folder  # noqa: PLC0415
        from pyvista.examples.downloads import download_file  # noqa: PLC0415
        from pyvista.examples.downloads import file_from_files  # noqa: PLC0415
        from pyvista.examples.examples import dir_path  # noqa: PLC0415

        if Path(path).is_absolute():
            # Absolute path must point to a built-in dataset
            assert Path(path).parent == Path(
                dir_path,
            ), 'Absolute path must point to a built-in dataset.'
            self._base_url = 'https://github.com/pyvista/pyvista/raw/main/pyvista/examples/'
            self._source_name = Path(path).name
            # the dataset is already downloaded (it's built-in)
            # so make download() simply return the local filepath
            self._download_func = lambda _: path
        else:
            # Relative path, use vars from downloads.py
            self._base_url = SOURCE
            self._download_func = download_file
            self._source_name = Path(path).name if Path(path).is_absolute() else path

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
            if os.path.isfile(self.path):
                try:
                    # Get file path
                    fullpath = file_from_files(target_file, self.path)
                except (FileNotFoundError, RuntimeError):
                    # Get folder path
                    fullpath = os.path.join(USER_DATA_PATH, path + '.unzip', target_file)
                    fullpath = fullpath if os.path.isdir(fullpath) else None
            # set the file path as the relative path of the target file if
            # the fullpath could not be resolved (i.e. not yet downloaded)
            self._path = target_file if fullpath is None else fullpath

    @property
    def source_name(self) -> str:
        return self._source_name

    @property
    def base_url(self) -> str:
        return self._base_url

    def download(self) -> str:
        path = self._download_func(self._source_name)
        assert os.path.isfile(path) or os.path.isdir(path)
        # Reset the path since the full path for archive files
        # isn't known until after downloading
        self._path = path
        return path


class _SingleFileDownloadableDatasetLoader(_SingleFileDatasetLoader, _DownloadableFile):
    """Wrap a single file which must first be downloaded and which can also be loaded.

    .. warning::

       ``download()`` should be called before accessing other attributes. Otherwise,
       calling ``load()`` or ``path`` may fail or produce unexpected results.

    """

    def __init__(  # noqa: PLR0917
        self,
        path: str,
        read_func: Callable[[str], DatasetType] | None = None,
        load_func: Callable[[DatasetType], DatasetType] | None = None,
        target_file: str | None = None,
    ):
        _SingleFileDatasetLoader.__init__(self, path, read_func=read_func, load_func=load_func)
        _DownloadableFile.__init__(self, path, target_file=target_file)


class _MultiFileDatasetLoader(_DatasetLoader, _MultiFilePropsProtocol):
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
        objects required for loading the dataset. Alternatively, a directory can be
        specified, in which case a separate single-file dataset loader is created
        for each file with a default reader.

    load_func
        Specify the function used to load the files. By default, :meth:`load()` is called
        on all the files (if loadable) and a tuple containing the loaded datasets is returned.

    """

    def __init__(
        self,
        files_func: str | Callable[[], Sequence[_SingleFileDatasetLoader | _DownloadableFile]],
        load_func: Callable[[Sequence[_SingleFileDatasetLoader]], Any] | None = None,
    ):
        self._files_func = files_func
        self._file_loaders_ = None
        if load_func is None:
            load_func = _load_as_dataset_or_multiblock

        _DatasetLoader.__init__(self, load_func)

    @property
    def _file_objects(self):
        if self._file_loaders_ is None and not isinstance(self._files_func, str):
            self._file_loaders_ = self._files_func()  # type: ignore[assignment]
        return self._file_loaders_

    @property
    def path(self) -> tuple[str, ...]:
        return tuple(_flatten_nested_sequence([file.path for file in self._file_objects]))

    @property
    def path_loadable(self) -> tuple[str, ...]:
        return tuple(
            file.path for file in self._file_objects if isinstance(file, _SingleFileDatasetLoader)
        )

    @property
    def _filesize_bytes(self) -> tuple[int, ...]:
        return tuple(
            _flatten_nested_sequence([file._filesize_bytes for file in self._file_objects]),
        )

    @property
    def _filesize_format(self) -> tuple[str, ...]:
        return tuple(_format_file_size(size) for size in self._filesize_bytes)

    @property
    def _total_size_bytes(self) -> int:
        return sum(file._total_size_bytes for file in self._file_objects)

    @property
    def total_size(self) -> str:
        return _format_file_size(self._total_size_bytes)

    @property
    def _reader(
        self,
    ) -> pv.BaseReader | tuple[pv.BaseReader | None, ...] | None:
        # TODO: return the actual reader used, and not just a lookup
        #       (this will require an update to the 'read_func' API)
        reader = _flatten_nested_sequence([file._reader for file in self._file_objects])
        # flatten in case any file objects themselves are multifiles
        reader_out: list[pv.BaseReader] = []
        for r in reader:
            reader_out.extend(r) if isinstance(r, Sequence) else reader_out.append(r)
        return tuple(reader_out)

    def load(self):
        return self._load_func(self._file_objects)


class _MultiFileDownloadableDatasetLoader(
    _MultiFileDatasetLoader,
    _Downloadable[tuple[str, ...]],
):
    """Wrap multiple files for downloading and loading."""

    @property
    def source_name(self) -> tuple[str, ...]:
        name = [file.source_name for file in self._file_objects if isinstance(file, _Downloadable)]
        return tuple(_flatten_nested_sequence(name))

    @property
    def base_url(self) -> tuple[str, ...]:
        url = [file.base_url for file in self._file_objects if isinstance(file, _Downloadable)]
        return tuple(_flatten_nested_sequence(url))

    def download(self) -> tuple[str, ...]:
        path = [file.download() for file in self._file_objects if isinstance(file, _Downloadable)]
        # flatten paths in case any loaders have multiple files
        path_out = _flatten_nested_sequence(path)
        assert all(os.path.isfile(p) or os.path.isdir(p) for p in path_out)
        return tuple(path_out)


_ScalarType = TypeVar('_ScalarType', int, str, pv.BaseReader)


def _flatten_nested_sequence(nested: Sequence[_ScalarType | Sequence[_ScalarType]]):
    """Flatten nested sequences of objects."""
    flat: list[_ScalarType] = []
    for item in nested:
        if isinstance(item, Sequence) and not isinstance(item, str):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def _download_dataset(
    dataset_loader: _SingleFileDownloadableDatasetLoader | _MultiFileDownloadableDatasetLoader,
    *,
    load: bool = True,
    metafiles: bool = False,
):
    """Download and load a dataset file or files.

    Parameters
    ----------
    dataset_loader
        SingleFile or MultiFile object(s) of the dataset(s) to download or load.

    load
        Read and load the file after downloading. When ``False``,
        return the path or paths to the example's file(s).

    metafiles
        When ``load`` is ``False``, set this value to ``True`` to
        return all files required to load the example, including any metafiles.
        If ``False``, only the paths of files which are explicitly loaded are
        returned. E.g if a file format uses two files to specify the header info
        and file data separately, setting ``metafiles=True`` will return a tuple
        with both file paths, whereas setting ``metafiles=False`` will only return
        the single path of the header file as a string.

    Returns
    -------
    Any
        Loaded dataset or path(s) to the example's files depending on the ``load``
        parameter. Dataset may be a texture, mesh, multiblock, array, tuple of meshes,
        or any other output loaded by the example.

    """
    # Download all files for the dataset, include any metafiles
    path = dataset_loader.download()

    # Exclude non-loadable metafiles from result (if any)
    if not metafiles and isinstance(dataset_loader, _MultiFileDownloadableDatasetLoader):
        path = dataset_loader.path_loadable
        # Return scalar if only one loadable file
        path = path[0] if len(path) == 1 else path

    return dataset_loader.load() if load else path


def _load_as_multiblock(
    files: Sequence[_SingleFileDatasetLoader | _MultiFileDatasetLoader],
    names: Sequence[str] | None = None,
) -> pv.MultiBlock:
    """Load multiple files as a MultiBlock.

    This function can be used as a loading function for :class:`MultiFileLoadable`
    If the use of the ``names`` parameter is needed, use :class:`functools.partial`
    to partially specify the names parameter before passing it as loading function.
    """
    multi = pv.MultiBlock()
    if names is None:
        # set names, use filename without ext by default or dirname
        paths = _flatten_nested_sequence(
            [file.path_loadable for file in files if isinstance(file, _DatasetLoader)],
        )
        paths = [Path(path) for path in paths]
        names = [
            path.name[: -len(get_ext(path.name))] if path.is_file() else path.name
            for path in paths
        ]

    for file, name in zip(files, names, strict=False):
        if not isinstance(file, _DatasetLoader):
            continue  # type: ignore[unreachable]
        loaded = file.load()
        assert isinstance(
            loaded,
            (pv.MultiBlock, pv.DataSet),
        ), (
            f'Only MultiBlock or DataSet objects can be loaded as a MultiBlock. '
            f"Got {type(loaded)}.'"
        )
        multi.append(loaded, name)
    return multi


def _load_as_cubemap(files: str | _SingleFile | Sequence[_SingleFile]) -> pv.Texture:
    """Load multiple files as a cubemap.

    Input may be a single directory with 6 cubemap files, or a sequence
    of 6 files
    """
    path = (
        files
        if isinstance(files, str)
        else (files.path if isinstance(files, _SingleFile) else [file.path for file in files])
    )

    return (
        pv.cubemap(path)
        if isinstance(files, str) and os.path.isdir(files)
        else pv.cubemap_from_filenames(path)
    )


def _load_as_dataset_or_multiblock(files):
    multiblock = _load_as_multiblock(files)
    return multiblock[0] if len(multiblock) == 1 else multiblock


def _load_and_merge(files: Sequence[_SingleFile]):
    """Load all loadable files as separate datasets and merge them."""
    loaded = [file.load() for file in files if isinstance(file, _DatasetLoader)]
    assert len(loaded) > 0
    return pv.merge(loaded)


def _get_file_or_folder_size(filepath) -> int:
    if os.path.isfile(filepath):
        return os.path.getsize(filepath)
    assert os.path.isdir(filepath), 'Expected a file or folder path.'
    all_filepaths = _get_all_nested_filepaths(filepath)
    return sum(os.path.getsize(file) for file in all_filepaths)


def _format_file_size(size: int) -> str:
    size_flt = float(size)
    for unit in ('B', 'KB', 'MB'):
        if round(size_flt * 10) / 10 < 1000.0:
            return f'{int(size_flt)} {unit}' if unit == 'B' else f'{size_flt:3.1f} {unit}'
        size_flt /= 1000.0
    return f'{size_flt:.1f} GB'


def _get_file_or_folder_ext(path: str):
    """Wrap the `get_ext` function to handle special cases for directories."""
    if os.path.isfile(path):
        return get_ext(path)
    assert os.path.isdir(path), 'Expected a file or folder path.'
    all_paths = _get_all_nested_filepaths(path)
    ext = [get_ext(file) for file in all_paths]
    assert len(ext) != 0, f'No files with extensions were found in"\n\t{path}'
    return ext


def _get_all_nested_filepaths(filepath, *, exclude_readme=True):
    """Walk through directory and get all file paths.

    Optionally exclude any readme files (if any).
    """
    assert os.path.isfile(filepath) or os.path.isdir(filepath)
    condition = lambda name: True if not exclude_readme else not name.lower().startswith('readme')
    return next(
        [os.path.join(path, name) for name in files if condition(name)]
        for path, _, files in os.walk(filepath)
    )


def _get_unique_extension(path: str | Sequence[str]):
    """Return a file extension or unique set of file extensions from a path or paths."""
    ext_set = set()
    fname_sequence = [path] if isinstance(path, str) else path

    # Add all file extensions to the set
    for file in fname_sequence:
        ext = _get_file_or_folder_ext(file)
        ext_set.add(ext) if isinstance(ext, str) else ext_set.update(ext)

    # Format output
    ext_output = tuple(ext_set)
    return ext_output[0] if len(ext_output) == 1 else tuple(sorted(ext_output))


def _get_unique_reader_type(
    reader: pv.BaseReader | tuple[pv.BaseReader | None, ...] | None,
) -> type[pv.BaseReader] | tuple[type[pv.BaseReader], ...] | None:
    """Return a reader type or tuple of unique reader types."""
    if reader is None or (isinstance(reader, Sequence) and all(r is None for r in reader)):
        return None
    reader_set: set[type[pv.BaseReader]] = set()
    reader_type = (
        [type(reader)]
        if not isinstance(reader, Sequence)
        else [type(r) for r in reader if r is not None]
    )

    # Add all reader types to the set
    reader_set.update(reader_type)

    # Format output
    reader_output = tuple(reader_set)
    return reader_output[0] if len(reader_output) == 1 else tuple(reader_output)


def _get_unique_dataset_type(
    dataset_iterable: tuple[DatasetObject, ...],
) -> DatasetType | tuple[DatasetType, ...]:
    """Return a dataset type or tuple of unique dataset types."""
    dataset_types: dict[DatasetType, None] = {}  # use dict as an ordered set
    for dataset in dataset_iterable:
        dataset_types[type(dataset)] = None
    output = tuple(dataset_types.keys())
    return output[0] if len(output) == 1 else output
