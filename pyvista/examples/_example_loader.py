"""Abstraction layer for downloading, reading, and loading example files."""

from __future__ import annotations

from abc import abstractmethod
from functools import partial
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
_FilenameType = TypeVar('_FilenameType', str, Tuple[str, ...], covariant=True)


class _Filename(Protocol[_FilenameType]):
    @property
    @abstractmethod
    def filename(self) -> _FilenameType: ...


@runtime_checkable
class _Downloadable(Protocol[_FilenameType]):
    @abstractmethod
    def download(self) -> _FilenameType: ...


@runtime_checkable
class _Loadable(Protocol):
    @abstractmethod
    def load(self) -> Any: ...


class _MultiFilename(_Filename[Tuple[str, ...]]): ...


class _SingleFilename(_Filename[str]): ...


class _SingleFileLoadable(_SingleFilename, _Loadable):
    """Wrap a single file for loading."""

    def __init__(
        self,
        filename: str,
        read_func: Optional[
            Callable[[str], pyvista.DataSet | pyvista.Texture | NumpyArray[Any]]
        ] = None,
        load_func: Optional[Callable[[pyvista.DataSet], Any]] = None,
    ):
        # TODO: Refactor filename stuff into parent class
        from pyvista.examples.downloads import USER_DATA_PATH

        self._filename = (
            filename if os.path.isabs(filename) else os.path.join(USER_DATA_PATH, filename)
        )
        ######
        self._read_func = pyvista.read if read_func is None else read_func
        self._load_func = load_func

    @property
    def filename(self):
        return self._filename

    def load(self):
        if self._load_func is None:
            return self._read_func(self.filename)
        return self._load_func(self._read_func(self.filename))


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

    def load(self):
        return self._load_func(self._files)


class _SingleFileDownloadable(_SingleFilename, _Downloadable[str]):
    """Wrap a single file which must downloaded."""

    def __init__(
        self,
        filename: str,
        target_file: Optional[str] = None,
    ):
        # TODO: Refactor this into SingleFile class
        from pyvista.examples.downloads import USER_DATA_PATH

        self._filename = (
            filename if os.path.isabs(filename) else os.path.join(USER_DATA_PATH, filename)
        )
        ######

        # TODO: Refactor this into Downloadable class
        from pyvista.examples.downloads import _download_archive_file_or_folder, download_file

        self._download_source = filename
        self._download_func = download_file
        if target_file is not None:
            # download from archive
            self._download_func = partial(_download_archive_file_or_folder, target_file=target_file)
            self._filename = target_file
        ######

    @property
    def filename(self):
        return self._filename

    def download(self) -> str:
        filename = self._download_func(self._download_source)
        assert os.path.isfile(filename)
        self._filename = filename
        return filename


class _SingleFileDownloadableLoadable(_SingleFileLoadable, _Downloadable[str]):
    """Wrap a single file which must downloaded and which can be loaded.

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
        from pyvista.examples.downloads import _download_archive_file_or_folder, download_file

        super().__init__(filename, read_func=read_func, load_func=load_func)
        self._download_source = filename
        self._download_func = download_file
        if target_file is not None:
            # download from archive
            self._download_func = partial(_download_archive_file_or_folder, target_file=target_file)
            self._filename = target_file

    def download(self) -> str:
        filename = self._download_func(self._download_source)
        assert os.path.isfile(filename) or os.path.isdir(filename)
        # Reset the filename since the full path for archive files
        # isn't known until after downloading
        self._filename = filename
        return filename


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
    """Load multiple files as a MultiBlock."""
    block = pyvista.MultiBlock()
    names = [os.path.basename(file.filename) for file in files] if names is None else names
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
    if isinstance(files, str) and os.path.isdir(files):
        return pyvista.cubemap(filename)
    return pyvista.cubemap_from_filenames(filename)


def _load_all(files: Sequence[_SingleFilename]):
    """Load all loadable files."""
    loaded = [file.load() for file in files if isinstance(file, _Loadable)]
    assert len(loaded) > 0
    return loaded[0] if len(loaded) == 1 else tuple(loaded)
