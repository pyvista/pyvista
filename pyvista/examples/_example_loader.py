"""Abstraction layer for downloading, reading, and loading example files."""

from __future__ import annotations

from abc import abstractmethod
import functools
import os
from typing import (
    Any,
    Callable,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

import pyvista
from pyvista.core._typing_core import NumpyArray

# The following classes define an API for working with either
# a single filename or multiple filenames

_T = TypeVar('_T')
_FilePropStrType = TypeVar('_FilePropStrType', str, Tuple[str, ...], covariant=True)


class _FileProps(Protocol[_FilePropStrType]):
    @property
    @abstractmethod
    def filename(self) -> _FilePropStrType:
        """Return the filename(s) of all files."""
        ...

    @property
    @abstractmethod
    def _filesize_bytes(self):
        """Return the file size(s) of all files in bytes."""
        ...

    @property
    @abstractmethod
    def _filesize_format(self) -> _FilePropStrType:
        """Return the formatted size of all file(s)."""
        ...

    @property
    @abstractmethod
    def total_size(self) -> str:
        """Return the total size of all files."""
        ...


@runtime_checkable
class _Downloadable(Protocol[_FilePropStrType]):
    @abstractmethod
    def download(self) -> _FilePropStrType: ...


@runtime_checkable
class _Loadable(Protocol):
    @abstractmethod
    def load(self) -> Any: ...


class _SingleFilename(_FileProps[str]):
    def __init__(self, filename):
        from pyvista.examples.downloads import USER_DATA_PATH

        self._filename = (
            filename if os.path.isabs(filename) else os.path.join(USER_DATA_PATH, filename)
        )

    @property
    def filename(self) -> str:
        return self._filename

    @property
    def _filesize_bytes(self) -> int:
        return _get_file_or_folder_size(self.filename)

    @property
    def _filesize_format(self) -> str:
        return _format_file_size(self._filesize_bytes)

    @property
    def total_size(self) -> str:
        return self._filesize_format


class _SingleFileLoadable(_SingleFilename, _Loadable):
    """Wrap a single file for loading.

    Specify the read function and/or load functions for reading and processing the
    dataset. The read function is called on the filename first, then, if a load
    function is specified, the load function is called on the output from the read
    function.

    Parameters
    ----------
    filename
        Path of the file to be loaded.

    read_func
        Specify the function used to read the file. Defaults to :func:`pyvista.read`.
        This can be used for customizing the reader's properties, or using another
        read function (e.g. :func:`pyvista.read_texture` for textures). The function
        must have the filename as the first argument and should return a dataset.
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
        filename: str,
        read_func: Optional[
            Callable[[str], pyvista.DataSet | pyvista.Texture | NumpyArray[Any]]
        ] = None,
        load_func: Optional[Callable[[pyvista.DataSet], Any]] = None,
    ):
        _SingleFilename.__init__(self, filename)
        self._read_func = pyvista.read if read_func is None else read_func
        self._load_func = load_func

    def load(self):
        if self._load_func is None:
            return self._read_func(self.filename)
        return self._load_func(self._read_func(self.filename))


class _SingleFileDownloadable(_SingleFilename, _Downloadable[str]):
    """Wrap a single file which must be downloaded.

    If downloading a file from an archive, set the `.zip` as the filename
    and set ``target_file`` as the file to extract. Set ``target_file=''``
    (empty string) to download the entire archive and return the directory
    path to the entire extracted archive.

    """

    def __init__(
        self,
        filename: str,
        target_file: Optional[str] = None,
    ):
        _SingleFilename.__init__(self, filename)

        from pyvista.examples.downloads import (
            USER_DATA_PATH,
            _download_archive_file_or_folder,
            download_file,
            file_from_files,
        )

        self._download_source = filename
        self._download_func = download_file
        if target_file is not None:
            # download from archive
            self._download_func = functools.partial(
                _download_archive_file_or_folder, target_file=target_file
            )
            # The filename currently points to the archive, not the target file itself
            # Try to resolve the full path to the target file (without downloading) if
            # the archive already exists in the cache
            fullpath = None
            if os.path.isfile(self.filename):
                try:
                    # Get file path
                    fullpath = file_from_files(target_file, self.filename)
                except (FileNotFoundError, RuntimeError):
                    # Get folder path
                    fullpath = os.path.join(USER_DATA_PATH, filename + '.unzip', target_file)
                    fullpath = fullpath if os.path.isdir(fullpath) else None
            # set the filename as the relative path of the target file if
            # the fullpath could not be resolved (i.e. not yet downloaded)
            self._filename = target_file if fullpath is None else fullpath

    def download(self) -> str:
        filename = self._download_func(self._download_source)
        assert os.path.isfile(filename)
        self._filename = filename
        return filename


class _SingleFileDownloadableLoadable(_SingleFileDownloadable, _SingleFileLoadable):
    """Wrap a single file which must first be downloaded and which can also be loaded.

    .. warning::

       ``download()`` should be called before accessing other attributes. Otherwise,
       calling ``load()`` or ``filename`` may fail or produce unexpected results.

    """

    def __init__(
        self,
        filename: str,
        read_func: Optional[
            Callable[[str], pyvista.DataSet | pyvista.Texture | NumpyArray[Any]]
        ] = None,
        load_func: Optional[Callable[[pyvista.DataSet], Any]] = None,
        target_file: Optional[str] = None,
    ):
        _SingleFileLoadable.__init__(self, filename, read_func=read_func, load_func=load_func)
        _SingleFileDownloadable.__init__(self, filename, target_file=target_file)

    def download(self) -> str:
        filename = self._download_func(self._download_source)
        assert os.path.isfile(filename) or os.path.isdir(filename)
        # Reset the filename since the full path for archive files
        # isn't known until after downloading
        self._filename = filename
        return filename


class _MultiFilename(_FileProps[Tuple[str, ...]]): ...


class _MultiFileLoadable(_MultiFilename, _Loadable):
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
        Specify the function which will return a sequence of :class:`_SingleFilename`
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
        self._files = files_func()
        if load_func is None:
            load_func = _load_all
        self._load_func = load_func

    @property
    def filename(self) -> Tuple[str, ...]:
        return tuple([file.filename for file in self._files])

    @property
    def filename_loadable(self) -> Tuple[str, ...]:
        return tuple(
            [file.filename for file in self._files if isinstance(file, _SingleFileLoadable)]
        )

    @property
    def _filesize_bytes(self) -> Tuple[int, ...]:
        return tuple([file._filesize_bytes for file in self._files])

    @property
    def _filesize_format(self) -> Tuple[str, ...]:
        return tuple([_format_file_size(size) for size in self._filesize_bytes])

    @property
    def total_size(self) -> str:
        return _format_file_size(sum(self._filesize_bytes))

    def load(self):
        return self._load_func(self._files)


class _MultiFileDownloadableLoadable(_MultiFileLoadable, _Downloadable[Tuple[str, ...]]):
    """Wrap multiple files for downloading and loading."""

    def download(self) -> Tuple[str, ...]:
        filename = [file.download() for file in self._files if isinstance(file, _Downloadable)]
        assert all(os.path.isfile(file) for file in filename)
        return tuple(filename)


def _download_example(
    example: Union[_SingleFileDownloadableLoadable, _MultiFileDownloadableLoadable],
    load: bool = True,
    metafiles: bool = False,
):
    """Download and load an example file or files.

    Parameters
    ----------
    example
        SingleFile or MultiFile object(s) to download or load.

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
        the single path of the data file as a string.

    Returns
    -------
    Any
        Loaded dataset or path(s) to the example's files depending on the ``load``
        parameter. Dataset may be a texture, mesh, multiblock, array, tuple of meshes,
        or any other output loaded by the example.

    """
    # Download all files for the dataset, include any metafiles
    filename = example.download()

    # Exclude non-loadable metafiles from result (if any)
    if not metafiles and isinstance(example, _MultiFileDownloadableLoadable):
        filename = example.filename_loadable
        # Return scalar if only one loadable file
        filename = filename[0] if len(filename) == 1 else filename

    return example.load() if load else filename


def _load_as_multiblock(
    files: Sequence[_SingleFilename], names: Optional[Sequence[str]] = None
) -> pyvista.MultiBlock:
    """Load multiple files as a MultiBlock.

    This function can be used as a loading function for :class:`MultiFileLoadable`
    If the use of the ``names`` parameter is needed, use :class:`functools.partial`
    to partially specify the names parameter before passing it as loading function.
    """
    block = pyvista.MultiBlock()
    names = (
        [os.path.splitext(os.path.basename(file.filename))[0] for file in files]
        if names is None
        else names
    )
    assert len(names) == len(files)
    [
        block.append(file.load(), name)
        for file, name in zip(files, names)
        if isinstance(file, _Loadable)
    ]
    return block


def _load_as_cubemap(
    files: Union[str, _SingleFilename, Sequence[_SingleFilename]]
) -> pyvista.Texture:
    """Load multiple files as a cubemap.

    Input may be a single directory with 6 cubemap files, or a sequence
    of 6 files
    """
    filename = (
        files
        if isinstance(files, str)
        else (
            files.filename
            if isinstance(files, _SingleFilename)
            else [file.filename for file in files]
        )
    )

    return (
        pyvista.cubemap(filename)
        if isinstance(files, str) and os.path.isdir(files)
        else pyvista.cubemap_from_filenames(filename)
    )


def _load_all(files: Sequence[_SingleFilename]):
    """Load all loadable files."""
    loaded = [file.load() for file in files if isinstance(file, _Loadable)]
    assert len(loaded) > 0
    return loaded[0] if len(loaded) == 1 else tuple(loaded)


def _get_file_or_folder_size(filepath) -> int:
    if os.path.isfile(filepath):
        return os.path.getsize(filepath)
    assert os.path.isdir(filepath), 'Expected a file or folder path.'
    all_filepaths = [
        [os.path.join(path, name) for name in files] for path, _, files in os.walk(filepath)
    ][0]
    return sum(os.path.getsize(path) for path in all_filepaths)


def _format_file_size(size):
    for unit in ("B", "KiB", "MiB"):
        if abs(size) < 1024.0:
            return f"{size:3.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GiB"
