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

# ruff: noqa: PTH102,PTH103,PTH107,PTH112,PTH113,PTH117,PTH118,PTH119,PTH122,PTH123,PTH202
from __future__ import annotations

from abc import abstractmethod
import functools
import os
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

import pyvista as pv
from pyvista.core._typing_core import NumpyArray
from pyvista.core.utilities.fileio import get_ext

DatasetType = Union[pv.DataSet, pv.DataSet, pv.Texture, NumpyArray[Any], pv.MultiBlock]
DatasetTypeType = Union[
    Type[pv.DataSet], Type[pv.Texture], Type[NumpyArray[Any]], Type[pv.MultiBlock]
]

# Define TypeVars for two main class definitions used by this module:
#   1. classes for single file inputs: T -> T
#   2. classes for multi-file inputs: (T, ...) -> (T, ...)
# Any properties with these typevars should have a one-to-one mapping for all files
_FilePropStrType_co = TypeVar('_FilePropStrType_co', str, Tuple[str, ...], covariant=True)
_FilePropIntType_co = TypeVar('_FilePropIntType_co', int, Tuple[int, ...], covariant=True)


class _FileProps(Protocol[_FilePropStrType_co, _FilePropIntType_co]):
    @property
    @abstractmethod
    def path(self) -> _FilePropStrType_co:
        """Return the path(s) of all files."""

    @property
    def num_files(self) -> int:
        """Return the number of files from path or paths.

        If a path is a folder, the number of files contained in the folder is returned.
        """
        path = [path] if isinstance(path := self.path, str) else path
        return sum([1 if os.path.isfile(p) else len(_get_all_nested_filepaths(p)) for p in path])

    @property
    def unique_extension(self) -> Union[str, Tuple[str, ...]]:
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
    ) -> Optional[Union[pv.BaseReader, Tuple[Optional[pv.BaseReader], ...]]]:
        """Return the base file reader(s) used to read the files."""

    @property
    def dataset(self) -> Optional[Union[DatasetType, Tuple[DatasetType, ...]]]:
        """Return dataset object(s)."""
        return None

    @property
    def unique_dataset_type(
        self,
    ) -> Optional[Union[DatasetTypeType, Tuple[DatasetTypeType, ...]]]:
        """Return unique dataset type(s) from all datasets."""
        return _get_unique_dataset_type(self.dataset)

    @property
    def unique_reader_type(
        self,
    ) -> Optional[Union[Type[pv.BaseReader], Tuple[Type[pv.BaseReader], ...]]]:
        """Return unique reader type(s) from all file readers."""
        return _get_unique_reader_type(self._reader)


@runtime_checkable
class _Downloadable(Protocol[_FilePropStrType_co]):
    """Class which implements a 'download' method."""

    @property
    @abstractmethod
    def download_url(self) -> _FilePropStrType_co:
        """Return the source of the download."""

    @property
    @abstractmethod
    def path(self) -> _FilePropStrType_co:
        """Return the file path of downloaded file."""

    @abstractmethod
    def download(self) -> _FilePropStrType_co:
        """Download and return the file path(s)."""


@runtime_checkable
class _Loadable(Protocol[_FilePropStrType_co]):
    """Class which loads a dataset from file."""

    @property
    @abstractmethod
    def dataset(self) -> Optional[Union[DatasetType, Tuple[DatasetType, ...]]]:
        """Return the loaded dataset object(s)."""

    @abstractmethod
    def load(self) -> DatasetType:
        """Load the dataset."""


class _SingleFile(_FileProps[str, int]):
    """Wrap a single file."""

    def __init__(self, path):
        from pyvista.examples.downloads import USER_DATA_PATH

        self._path = path if os.path.isabs(path) else os.path.join(USER_DATA_PATH, path)

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
    def _reader(self) -> Optional[pv.BaseReader]:
        return None


class _SingleFileLoadable(_SingleFile, _Loadable[str]):
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
        read_func: Optional[Callable[[str], DatasetType]] = None,
        load_func: Optional[Callable[[pv.DataSet], Any]] = None,
    ):
        _SingleFile.__init__(self, path)
        self._read_func = pv.read if read_func is None else read_func
        self._load_func = load_func
        self._dataset = None

    @property
    def dataset(self) -> Optional[DatasetType]:
        return self._dataset

    @property
    def _reader(self) -> Optional[pv.BaseReader]:
        # TODO: return the actual reader used, and not just a lookup
        #       (this will require an update to the 'read_func' API)
        try:
            return pv.get_reader(self.path)
        except ValueError:
            # Cannot be read directly (requires custom reader)
            return None

    def load(self):
        self._dataset = (
            self._read_func(self.path)
            if self._load_func is None
            else self._load_func(self._read_func(self.path))
        )
        return self.dataset


class _SingleFileDownloadable(_SingleFile, _Downloadable[str]):
    """Wrap a single file which must be downloaded.

    If downloading a file from an archive, set the filepath of the zip as
    ``path`` and set ``target_file`` as the file to extract. Set ``target_file=''``
    (empty string) to download the entire archive and return the directory
    path to the entire extracted archive.

    """

    def __init__(
        self,
        path: str,
        target_file: Optional[str] = None,
    ):
        _SingleFile.__init__(self, path)

        from pyvista.examples.downloads import (
            USER_DATA_PATH,
            _download_archive_file_or_folder,
            download_file,
            file_from_files,
        )

        self._download_source = path
        self._download_func = download_file
        if target_file is not None:
            # download from archive
            self._download_func = functools.partial(
                _download_archive_file_or_folder, target_file=target_file
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
    def download_url(self) -> str:
        from pyvista.examples.downloads import SOURCE

        return os.path.join(SOURCE, self._download_source)

    def download(self) -> str:
        path = self._download_func(self._download_source)
        assert os.path.isfile(path) or os.path.isdir(path)
        # Reset the path since the full path for archive files
        # isn't known until after downloading
        self._path = path
        return path


class _SingleFileDownloadableLoadable(_SingleFileDownloadable, _SingleFileLoadable):
    """Wrap a single file which must first be downloaded and which can also be loaded.

    .. warning::

       ``download()`` should be called before accessing other attributes. Otherwise,
       calling ``load()`` or ``path`` may fail or produce unexpected results.

    """

    def __init__(
        self,
        path: str,
        read_func: Optional[Callable[[str], DatasetType]] = None,
        load_func: Optional[Callable[[pv.DataSet], Any]] = None,
        target_file: Optional[str] = None,
    ):
        _SingleFileLoadable.__init__(self, path, read_func=read_func, load_func=load_func)
        _SingleFileDownloadable.__init__(self, path, target_file=target_file)


class _MultiFile(_FileProps[Tuple[str, ...], Tuple[int, ...]]):
    """Wrap multiple files."""


class _MultiFileLoadable(_MultiFile, _Loadable[Tuple[str, ...]]):
    """Wrap multiple files for loading.

    Some use cases for loading multi-file examples include:

    1. Multiple input files, and each file is read/loaded independently
       E.g.: loading two separate datasets for the example
       See ``download_bolt_nut`` for a reference implementation.

    2. Multiple input files, but only one is read or loaded directly
       E.g.: loading a single dataset from a file format where data and metadata are
       stored in separate files, such as ``.raw`` and ``.mhd``.
       See ``download_frog`` for a reference implementation.

    3. Multiple input files, all of which make up part of the loaded dataset
       E.g.: loading six separate image files for cubemaps
       See ``download_sky_box_cube_map`` for a reference implementation.

    Parameters
    ----------
    files_func
        Specify the function which will return a sequence of :class:`_SingleFile`
        objects which are used by an example.

    load_func
        Specify the function used to load the files. By default, :meth:`load()` is called
        on all the files (if loadable) and a tuple containing the loaded datasets is returned.

    """

    def __init__(
        self,
        files_func: Callable[[], Sequence[Union[_SingleFileLoadable, _SingleFileDownloadable]]],
        load_func: Optional[Callable[[Sequence[_SingleFileLoadable]], Any]] = None,
    ):
        self._files_func = files_func
        self._file_loaders_ = None
        if load_func is None:
            load_func = _load_all
        self._load_func = load_func
        self._dataset = None

    @property
    def _file_objects(self):
        if self._file_loaders_ is None:
            self._file_loaders_ = self._files_func()
        return self._file_loaders_

    @property
    def path(self) -> Tuple[str, ...]:
        return tuple(_flatten_path([file.path for file in self._file_objects]))

    @property
    def path_loadable(self) -> Tuple[str, ...]:
        return tuple(
            [file.path for file in self._file_objects if isinstance(file, _SingleFileLoadable)]
        )

    @property
    def _filesize_bytes(self) -> Tuple[int, ...]:
        return tuple([file._filesize_bytes for file in self._file_objects])

    @property
    def _filesize_format(self) -> Tuple[str, ...]:
        return tuple([_format_file_size(size) for size in self._filesize_bytes])

    @property
    def _total_size_bytes(self) -> int:
        return sum([file._total_size_bytes for file in self._file_objects])

    @property
    def total_size(self) -> str:
        return _format_file_size(self._total_size_bytes)

    @property
    def _reader(
        self,
    ) -> Optional[Union[pv.BaseReader, Tuple[Optional[pv.BaseReader], ...]]]:
        # TODO: return the actual reader used, and not just a lookup
        #       (this will require an update to the 'read_func' API)
        reader = [file._reader for file in self._file_objects]
        # flatten in case any file objects themselves are multifiles
        reader_out: List[pv.BaseReader] = []
        for r in reader:
            reader_out.extend(r) if isinstance(r, Sequence) else reader_out.append(r)
        return tuple(reader_out)

    @property
    def dataset(self) -> Optional[DatasetType]:
        return self._dataset

    def load(self):
        self._dataset = self._load_func(self._file_objects)
        return self.dataset


class _MultiFileDownloadableLoadable(_MultiFileLoadable, _Downloadable[Tuple[str, ...]]):
    """Wrap multiple files for downloading and loading."""

    @property
    def download_url(self) -> Tuple[str, ...]:
        return tuple(
            [file.download_url for file in self._file_objects if isinstance(file, _Downloadable)]
        )

    def download(self) -> Tuple[str, ...]:
        path = [file.download() for file in self._file_objects if isinstance(file, _Downloadable)]
        # flatten paths in case any loaders have multiple files
        path_out = _flatten_path(path)
        assert all(os.path.isfile(p) or os.path.isdir(p) for p in path_out)
        return tuple(path_out)


def _flatten_path(
    path: Union[
        str,
        Union[List[str], Tuple[str, ...]],
        Sequence[Union[str, Union[List[str], Tuple[str, ...]]]],
    ]
):
    """Flatten path or nested sequences of paths and return a single path or a list of paths."""
    if isinstance(path, str):
        return path
    else:
        path_out = []
        for p in path:
            path_out.append(p) if isinstance(p, str) else path_out.extend(p)
        return path_out


def _download_dataset(
    dataset: Union[_SingleFileDownloadableLoadable, _MultiFileDownloadableLoadable],
    load: bool = True,
    metafiles: bool = False,
):
    """Download and load a dataset file or files.

    Parameters
    ----------
    dataset
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
    path = dataset.download()

    # Exclude non-loadable metafiles from result (if any)
    if not metafiles and isinstance(dataset, _MultiFileDownloadableLoadable):
        path = dataset.path_loadable
        # Return scalar if only one loadable file
        path = path[0] if len(path) == 1 else path

    return dataset.load() if load else path


def _load_as_multiblock(
    files: Sequence[_SingleFile], names: Optional[Sequence[str]] = None
) -> pv.MultiBlock:
    """Load multiple files as a MultiBlock.

    This function can be used as a loading function for :class:`MultiFileLoadable`
    If the use of the ``names`` parameter is needed, use :class:`functools.partial`
    to partially specify the names parameter before passing it as loading function.
    """
    block = pv.MultiBlock()
    names = (
        [os.path.splitext(os.path.basename(file.path))[0] for file in files]
        if names is None
        else names
    )
    assert len(names) == len(files)
    [
        block.append(file.load(), name)  # type: ignore[arg-type]
        for file, name in zip(files, names)
        if isinstance(file, _Loadable)
    ]
    return block


def _load_as_cubemap(files: Union[str, _SingleFile, Sequence[_SingleFile]]) -> pv.Texture:
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


def _load_all(files: Sequence[_SingleFile]):
    """Load all loadable files."""
    loaded = [file.load() for file in files if isinstance(file, _Loadable)]
    assert len(loaded) > 0
    return loaded[0] if len(loaded) == 1 else tuple(loaded)


def _load_and_merge(files: Sequence[_SingleFile]):
    return pv.merge(_load_all(files))


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
            return f"{int(size_flt)} {unit}" if unit == 'B' else f"{size_flt:3.1f} {unit}"
        size_flt /= 1000.0
    return f"{size_flt:.1f} GB"


def _get_file_or_folder_ext(path: str):
    """Wrap the `get_ext` function to handle special cases for directories."""
    if os.path.isfile(path):
        return get_ext(path)
    assert os.path.isdir(path), 'Expected a file or folder path.'
    all_paths = _get_all_nested_filepaths(path)
    ext = [get_ext(file) for file in all_paths]
    assert len(ext) != 0, f'No files with extensions were found in"\n\t{path}'
    return ext


def _get_all_nested_filepaths(filepath, exclude_readme=True):
    """Walk through directory and get all file paths.

    Optionally exclude any readme files (if any).
    """
    condition = lambda name: True if not exclude_readme else not name.lower().startswith('readme')
    return [
        [os.path.join(path, name) for name in files if condition(name)]
        for path, _, files in os.walk(filepath)
    ][0]


def _get_unique_extension(path: Union[str, Sequence[str]]):
    """Return a file extension or unique set of file extensions from a path or paths."""
    ext_set = set()
    fname_sequence = [path] if isinstance(path, str) else path

    # Add all file extensions to the set
    for file in fname_sequence:
        ext = _get_file_or_folder_ext(file)
        ext_set.add(ext) if isinstance(ext, str) else ext_set.update(ext)

    # Format output
    ext_output = list(ext_set)
    if len(ext_output) == 1:
        return ext_output[0]
    elif (
        len(ext_output) == len(fname_sequence) and isinstance(path, str) and not os.path.isdir(path)
    ):
        # If num extensions matches num files, make
        # sure the extension order matches the fname order
        return tuple([get_ext(name) for name in path])
    else:
        return tuple(sorted(ext_output))


def _get_unique_reader_type(
    reader: Optional[Union[pv.BaseReader, Tuple[Optional[pv.BaseReader], ...]]]
) -> Optional[Union[Type[pv.BaseReader], Tuple[Type[pv.BaseReader], ...]]]:
    """Return a reader type or tuple of unique reader types."""
    if reader is None or (isinstance(reader, Sequence) and all(r is None for r in reader)):
        return None
    reader_set: Set[Type[pv.BaseReader]] = set()
    reader_type = (
        [type(reader)]
        if not isinstance(reader, Sequence)
        else [type(r) for r in reader if r is not None]
    )

    # Add all reader types to the set and exclude any 'None' types
    reader_set.update(reader_type)
    reader_output = tuple(reader_set)

    # Format output
    if len(reader_output) == 1:
        return reader_output[0]
    elif len(reader_type) == len(reader_output):
        # If num readers matches num files, make
        # sure the reader order matches the file order
        return tuple(reader_type)
    else:
        return tuple(reader_output)


def _get_unique_dataset_type(
    dataset: Optional[Union[DatasetType, Tuple[DatasetType, ...]]],
) -> Optional[Union[DatasetTypeType, Tuple[DatasetTypeType, ...]]]:
    """Return a dataset type or tuple of unique dataset types."""
    if dataset is None or (isinstance(dataset, Sequence) and all(d is None for d in dataset)):
        return None
    dataset_set: Set[DatasetTypeType] = set()
    if isinstance(dataset, Sequence):
        # Include all sub-dataset types from tuple[datataset, ...] or MultiBlock[dataset, ...]
        [dataset_set.add(type(d)) for d in dataset if d is not None]  # type: ignore[func-returns-value]
    else:
        dataset_set.add(type(dataset))

    # Add MultiBlock type itself to output and make sure it's the first entry
    if isinstance(dataset, pv.MultiBlock):
        dataset_set.add(pv.MultiBlock)
        dataset_set.discard(pv.MultiBlock)
        dataset_list = list(dataset_set)
        dataset_list.insert(0, pv.MultiBlock)
        dataset_output = tuple(dataset_list)
    else:
        dataset_output = tuple(dataset_set)

    return dataset_output[0] if len(dataset_output) == 1 else dataset_output
