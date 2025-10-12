"""Downloadable datasets collected from various sources.

Once downloaded, these datasets are stored locally allowing for the
rapid reuse of these datasets.

Files are all hosted in https://github.com/pyvista/vtk-data/ and are downloaded
using the ``download_file`` function. If you add a file to the example data
repository, you should add a ``download-<dataset>`` method here which will
rendered on this page.

See the :ref:`dataset_gallery` for detailed information about the datasets,
including file and dataset metadata.

Examples
--------
>>> from pyvista import examples
>>> mesh = examples.download_saddle_surface()
>>> mesh.plot()

"""

from __future__ import annotations

import functools
import importlib.util
import logging
import os
from pathlib import Path
from pathlib import PureWindowsPath
import shutil
import sys
from typing import cast
import warnings

import numpy as np
import pooch
from pooch import Unzip
from pooch.utils import get_logger

import pyvista
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core import _vtk_core as _vtk
from pyvista.core.utilities.fileio import get_ext
from pyvista.core.utilities.fileio import read
from pyvista.core.utilities.fileio import read_texture
from pyvista.examples._dataset_loader import _download_dataset
from pyvista.examples._dataset_loader import _DownloadableFile
from pyvista.examples._dataset_loader import _load_as_cubemap
from pyvista.examples._dataset_loader import _load_as_multiblock
from pyvista.examples._dataset_loader import _MultiFileDownloadableDatasetLoader
from pyvista.examples._dataset_loader import _SingleFileDownloadableDatasetLoader

# disable pooch verbose logging
POOCH_LOGGER = get_logger()
POOCH_LOGGER.setLevel(logging.CRITICAL)


CACHE_VERSION = 3

_USERDATA_PATH_VARNAME = 'PYVISTA_USERDATA_PATH'
_VTK_DATA_VARNAME = 'PYVISTA_VTK_DATA'

_DEFAULT_USER_DATA_PATH = str(pooch.os_cache(f'pyvista_{CACHE_VERSION}'))
_DEFAULT_VTK_DATA_SOURCE = 'https://github.com/pyvista/vtk-data/raw/master/Data/'


def _warn_invalid_dir_not_used(path, env_var):
    msg = f'The given {env_var} is not a valid directory and will not be used:\n{path.as_posix()}'
    warnings.warn(msg, stacklevel=2)


def _get_vtk_data_source() -> tuple[str, bool]:
    # If available, a local vtk-data instance will be used for examples
    # Set default output
    source = _DEFAULT_VTK_DATA_SOURCE
    file_cache = False
    if _VTK_DATA_VARNAME in os.environ:
        path = Path(os.environ[_VTK_DATA_VARNAME])
        if not path.is_dir():
            _warn_invalid_dir_not_used(path, _VTK_DATA_VARNAME)
        else:
            if path.name != 'Data':
                # append 'Data' if user does not provide it
                path = path / 'Data'

            # pooch assumes this is a URL so we have to take care of this
            path_str = path.as_posix()
            if not path_str.endswith('/'):
                path_str += '/'

            source = path_str
            file_cache = True
    return source, file_cache


def _get_user_data_path() -> str:
    # Allow user to override the local path
    # Set default output
    output_path = _DEFAULT_USER_DATA_PATH
    if _USERDATA_PATH_VARNAME in os.environ:
        path = Path(os.environ[_USERDATA_PATH_VARNAME])
        if not path.is_dir():
            _warn_invalid_dir_not_used(path, _USERDATA_PATH_VARNAME)
        else:
            # Use user-specified path
            output_path = str(path)
    return output_path


def _warn_if_path_not_accessible(path: str | Path, msg: str):
    # Provide helpful message if pooch path is inaccessible
    try:
        if not Path(path).is_dir():
            Path(path).mkdir(exist_ok=True, parents=True)
            if not os.access(path, os.W_OK):  # pragma: no cover
                raise OSError
    except (PermissionError, OSError):
        # Warn, don't raise just in case there's an environment issue.
        msg = f'Unable to access path: {path}\n{msg}'
        warnings.warn(msg, stacklevel=2)


SOURCE, _FILE_CACHE = _get_vtk_data_source()
USER_DATA_PATH = _get_user_data_path()

_user_data_path_warn_msg = (
    f'Manually specify the PyVista examples cache with the '
    f'{_USERDATA_PATH_VARNAME} environment variable.'
)
_warn_if_path_not_accessible(USER_DATA_PATH, _user_data_path_warn_msg)

# Note that our fetcher doesn't have a registry (or we have an empty registry)
# with hashes because we don't want to have to add in all of them individually
# to the registry since we're not (at the moment) concerned about hashes.
FETCHER = pooch.create(
    path=USER_DATA_PATH,
    base_url=SOURCE,
    registry={},
    retry_if_failed=3,
)


def file_from_files(target_path, fnames):
    """Return the full path of a single file within a list of files.

    Parameters
    ----------
    target_path : str
        Path of the file to match the end of. If you need to match a file
        relative to the root directory of the archive, start the path with
        ``"unzip"``. Path must be a posix-like path.

    fnames : list
        List of filenames.

    Returns
    -------
    str
        Entry in ``fnames`` matching ``filename``.

    """
    found_fnames = []
    for fname in fnames:
        # always convert windows paths
        posix_fname = PureWindowsPath(fname).as_posix() if os.name == 'nt' else fname
        # ignore mac hidden directories
        if '/__MACOSX/' in posix_fname:  # pragma: no cover
            continue
        if posix_fname.endswith(target_path):
            found_fnames.append(posix_fname)

    if len(found_fnames) == 1:
        return found_fnames[0]

    if len(found_fnames) > 1:
        files_str = '\n'.join(found_fnames)
        msg = f'Ambiguous "{target_path}". Multiple matches found:\n{files_str}'
        raise RuntimeError(msg)

    files_str = '\n'.join(fnames)
    msg = f'Missing "{target_path}" from archive. Archive contains:\n{files_str}'
    raise FileNotFoundError(msg)


def _file_copier(input_file, output_file, *_, **__):
    """Copy a file from a local directory to the output path."""
    if not Path(input_file).is_file():
        msg = f"'{input_file}' not found within PYVISTA_VTK_DATA '{SOURCE}'"
        raise FileNotFoundError(msg)
    shutil.copy(input_file, output_file)


def download_file(filename):
    """Download a single file from the PyVista vtk-data repository.

    You can add an example file at `pyvista/vtk_data
    <https://github.com/pyvista/vtk-data>`_.

    Parameters
    ----------
    filename : str
        Filename relative to the ``Data`` directory.

    Returns
    -------
    str | list
        A single path if the file is not an archive. A ``list`` of paths if the
        file is an archive.

    Examples
    --------
    Download the ``'puppy.jpg'`` image.

    >>> from pyvista import examples
    >>> path = examples.download_file('puppy.jpg')  # doctest:+SKIP
    >>> path  # doctest:+SKIP
    '/home/user/.cache/pyvista_3/puppy.jpg'

    """
    try:  # should the file already exist within fetcher's registry
        return _download_file(filename)
    except ValueError:  # otherwise simply add the file to the registry
        FETCHER.registry[filename] = None
        return _download_file(filename)


def _download_file(filename):
    """Download a file using pooch."""
    return FETCHER.fetch(
        filename,
        processor=Unzip() if filename.endswith('.zip') else None,
        downloader=_file_copier if _FILE_CACHE else None,
    )


def _download_archive(filename, target_file=None):
    """Download an archive.

    Return the path to a single file when set.

    Parameters
    ----------
    filename : str
        Relative path to the archive file. The entire archive will be
        downloaded and unarchived.

    target_file : str, optional
        Target file to return within the archive.

    Returns
    -------
    list | str
        List of files when ``target_file`` is ``None``. Otherwise, a single path.

    """
    fnames = download_file(filename)
    if target_file is not None:
        return file_from_files(target_file, fnames)
    return fnames


def _download_archive_file_or_folder(filename, target_file=None):
    """Download an archive.

    This function is similar to _download_archive, but also allows
    setting `target_file` as a folder. The target folder path must be
    fully specified relative to the root path of the archive.

    Set ``target_file=''`` (empty string) to download the entire
    archive and return the directory path to the entire extracted
    archive.

    """
    try:
        # Return file(s)
        return _download_archive(filename, target_file=target_file)
    except (FileNotFoundError, RuntimeError):
        pass
    # Return folder, or re-raise error by calling function again
    folder = str(Path(USER_DATA_PATH) / (filename + '.unzip') / target_file)
    return (
        folder if Path(folder).is_dir() else _download_archive(filename, target_file=target_file)
    )


def delete_downloads():
    """Delete all downloaded examples to free space or update the files.

    Examples
    --------
    Delete all local downloads.

    >>> from pyvista import examples
    >>> examples.delete_downloads()  # doctest:+SKIP

    """
    if Path(USER_DATA_PATH).is_dir():
        shutil.rmtree(USER_DATA_PATH)
    Path(USER_DATA_PATH).mkdir()


def _download_and_read(filename, *, texture=False, file_format=None, load=True):
    """Download and read a file.

    Parameters
    ----------
    filename : str
        Path to the filename. This cannot be a zip file.

    texture : bool, default: False
        ``True`` when file being read is a texture.

    file_format : str, optional
        Override the file format with a different extension.

    load : bool, default: True
        Read the file. When ``False``, return the path to the
        file.

    Returns
    -------
    pyvista.DataSet | str
        Dataset or path to the file depending on the ``load`` parameter.

    """
    if get_ext(filename) == '.zip':  # pragma: no cover
        msg = 'Cannot download and read an archive file'
        raise ValueError(msg)

    saved_file = download_file(filename)
    if not load:
        return saved_file
    if texture:
        return read_texture(saved_file)
    return read(saved_file, file_format=file_format)


@_deprecate_positional_args
def download_masonry_texture(load=True):  # noqa: FBT002
    """Download masonry texture.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.DataSet | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Create plot the masonry testure on a surface.

    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> texture = examples.download_masonry_texture()
    >>> surf = pv.Cylinder()
    >>> surf.plot(texture=texture)

    .. seealso::

        :ref:`Masonry Texture Dataset <masonry_texture_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`texture_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_masonry_texture, load=load)


_dataset_masonry_texture = _SingleFileDownloadableDatasetLoader(
    'masonry.bmp',
    read_func=read_texture,  # type: ignore[arg-type]
)


@_deprecate_positional_args
def download_usa_texture(load=True):  # noqa: FBT002
    """Download USA texture.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.DataSet | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> dataset = examples.download_usa_texture()
    >>> dataset.plot(cpos='xy')

    .. seealso::

        :ref:`Usa Texture Dataset <usa_texture_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Usa Dataset <usa_dataset>`

    """
    return _download_dataset(_dataset_usa_texture, load=load)


_dataset_usa_texture = _SingleFileDownloadableDatasetLoader(
    'usa_image.jpg',
    read_func=read_texture,  # type: ignore[arg-type]
)


@_deprecate_positional_args
def download_puppy_texture(load=True):  # noqa: FBT002
    """Download puppy texture.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.DataSet | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_puppy_texture()
    >>> dataset.plot(cpos='xy')

    .. seealso::

        :ref:`Puppy Texture Dataset <puppy_texture_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Puppy Dataset <puppy_dataset>`

        :ref:`texture_example`
            Example which uses this dataset.

    """
    return _download_dataset(_dataset_puppy_texture, load=load)


_dataset_puppy_texture = _SingleFileDownloadableDatasetLoader('puppy.jpg', read_func=read_texture)  # type: ignore[arg-type]


@_deprecate_positional_args
def download_puppy(load=True):  # noqa: FBT002
    """Download puppy dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ImageData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_puppy()
    >>> dataset.plot(cpos='xy', rgba=True)

    .. seealso::

        :ref:`Puppy Dataset <puppy_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Puppy Texture Dataset <puppy_texture_dataset>`

        :ref:`read_image_example`

    """
    return _download_dataset(_dataset_puppy, load=load)


_dataset_puppy = _SingleFileDownloadableDatasetLoader('puppy.jpg')


@_deprecate_positional_args
def download_usa(load=True):  # noqa: FBT002
    """Download usa dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.DataSet | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_usa()
    >>> dataset.plot(style='wireframe', cpos='xy')

    .. seealso::

        :ref:`Usa Dataset <usa_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Usa Texture Dataset <usa_texture_dataset>`

    """
    return _download_dataset(_dataset_usa, load=load)


_dataset_usa = _SingleFileDownloadableDatasetLoader('usa.vtk')


@_deprecate_positional_args
def download_st_helens(load=True):  # noqa: FBT002
    """Download Saint Helens dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ImageData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_st_helens()
    >>> dataset.plot(cmap='gist_earth')

    .. seealso::

        :ref:`St Helens Dataset <st_helens_dataset>`
            See this dataset in the Dataset Gallery for more info.

        This dataset is used in the following examples:

        * :ref:`colormap_example`
        * :ref:`lighting_mesh_example`
        * :ref:`opacity_example`
        * :ref:`orbit_example`
        * :ref:`plot_over_line_example`
        * :ref:`plotter_lighting_example`
        * :ref:`themes_example`

    """
    return _download_dataset(_dataset_st_helens, load=load)


_dataset_st_helens = _SingleFileDownloadableDatasetLoader('SainteHelens.dem')


@_deprecate_positional_args
def download_bunny(load=True):  # noqa: FBT002
    """Download bunny dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_bunny()
    >>> dataset.plot(cpos='xy')

    .. seealso::

        :ref:`Bunny Dataset <bunny_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Bunny Coarse Dataset <bunny_coarse_dataset>`

        This dataset is used in the following examples:

        * :ref:`read_file_example`
        * :ref:`clip_with_surface_example`
        * :ref:`extract_edges_example`
        * :ref:`subdivide_example`
        * :ref:`silhouette_example`
        * :ref:`light_types_example`

    """
    return _download_dataset(_dataset_bunny, load=load)


_dataset_bunny = _SingleFileDownloadableDatasetLoader('bunny.ply')


@_deprecate_positional_args
def download_bunny_coarse(load=True):  # noqa: FBT002
    """Download coarse bunny dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_bunny_coarse()
    >>> dataset.plot(cpos='xy')

    .. seealso::

        :ref:`Bunny Coarse Dataset <bunny_coarse_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Bunny Dataset <bunny_dataset>`

        This dataset is used in the following examples:

        * :ref:`read_file_example`
        * :ref:`clip_with_surface_example`
        * :ref:`subdivide_example`

    """
    return _download_dataset(_dataset_bunny_coarse, load=load)


def _bunny_coarse_load_func(mesh):
    mesh.verts = np.array([], dtype=np.int32)
    return mesh


_dataset_bunny_coarse = _SingleFileDownloadableDatasetLoader(
    'Bunny.vtp',
    load_func=_bunny_coarse_load_func,
)


@_deprecate_positional_args
def download_cow(load=True):  # noqa: FBT002
    """Download cow dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_cow()
    >>> dataset.plot(cpos='xy')

    .. seealso::

        :ref:`Cow Dataset <cow_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Cow Head Dataset <cow_head_dataset>`

        This dataset is used in the following examples:

        * :ref:`extract_edges_example`
        * :ref:`mesh_quality_example`
        * :ref:`rotate_example`
        * :ref:`linked_views_example`
        * :ref:`light_actors_example`

    """
    return _download_dataset(_dataset_cow, load=load)


_dataset_cow = _SingleFileDownloadableDatasetLoader('cow.vtp')


@_deprecate_positional_args
def download_cow_head(load=True):  # noqa: FBT002
    """Download cow head dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_cow_head()
    >>> dataset.plot(cpos='xy')

    .. seealso::

        :ref:`Cow Head Dataset <cow_head_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Cow Dataset <cow_dataset>`

    """
    return _download_dataset(_dataset_cow_head, load=load)


_dataset_cow_head = _SingleFileDownloadableDatasetLoader('cowHead.vtp')


@_deprecate_positional_args
def download_faults(load=True):  # noqa: FBT002
    """Download faults dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_faults()
    >>> dataset.plot(line_width=4)

    .. seealso::

        :ref:`Faults Dataset <faults_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_faults, load=load)


_dataset_faults = _SingleFileDownloadableDatasetLoader('faults.vtk')


@_deprecate_positional_args
def download_tensors(load=True):  # noqa: FBT002
    """Download tensors dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_tensors()
    >>> dataset.plot()

    .. seealso::

        :ref:`Tensors Dataset <tensors_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_tensors, load=load)


_dataset_tensors = _SingleFileDownloadableDatasetLoader('tensors.vtk')


@_deprecate_positional_args
def download_head(load=True):  # noqa: FBT002
    """Download head dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ImageData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> dataset = examples.download_head()
    >>> pl = pv.Plotter()
    >>> _ = pl.add_volume(dataset, cmap='cool', opacity='sigmoid_6')
    >>> pl.camera_position = [
    ...     (-228.0, -418.0, -158.0),
    ...     (94.0, 122.0, 82.0),
    ...     (-0.2, -0.3, 0.9),
    ... ]
    >>> pl.show()

    .. seealso::

        :ref:`Head Dataset <head_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Head 2 Dataset <head_2_dataset>`

        :ref:`medical_dataset_gallery`
            Browse other medical datasets.

        :ref:`volume_rendering_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_head, load=load)


def _head_files_func():
    # Multiple files needed for read, but only one gets loaded
    head_raw = _DownloadableFile('HeadMRVolume.raw')
    head_mhd = _SingleFileDownloadableDatasetLoader('HeadMRVolume.mhd')
    return head_mhd, head_raw


_dataset_head = _MultiFileDownloadableDatasetLoader(_head_files_func)


@_deprecate_positional_args
def download_head_2(load=True):  # noqa: FBT002
    """Download head dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ImageData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> dataset = examples.download_head_2()
    >>> pl = pv.Plotter()
    >>> _ = pl.add_volume(dataset, cmap='cool', opacity='sigmoid_6')
    >>> pl.show()

    .. seealso::

        :ref:`Head 2 Dataset <head_2_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Head Dataset <head_dataset>`

        :ref:`medical_dataset_gallery`
            Browse other medical datasets.

    """
    return _download_dataset(_dataset_head_2, load=load)


_dataset_head_2 = _SingleFileDownloadableDatasetLoader('head.vti')


@_deprecate_positional_args
def download_bolt_nut(load=True):  # noqa: FBT002
    """Download bolt nut dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.MultiBlock or tuple
        DataSet or tuple of filenames depending on ``load``.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> dataset = examples.download_bolt_nut()
    >>> pl = pv.Plotter()
    >>> _ = pl.add_volume(
    ...     dataset,
    ...     cmap='coolwarm',
    ...     opacity='sigmoid_5',
    ...     show_scalar_bar=False,
    ... )
    >>> pl.camera_position = [
    ...     (194.6, -141.8, 182.0),
    ...     (34.5, 61.0, 32.5),
    ...     (-0.229, 0.45, 0.86),
    ... ]
    >>> pl.show()

    .. seealso::

        :ref:`Bolt Nut Dataset <bolt_nut_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`volume_rendering_example`
            Example which uses this dataset.

    """
    return _download_dataset(_dataset_bolt_nut, load=load)


def _bolt_nut_files_func():
    # Multiple mesh files are loaded for this example
    bolt = _SingleFileDownloadableDatasetLoader('bolt.slc')
    nut = _SingleFileDownloadableDatasetLoader('nut.slc')
    return bolt, nut


_dataset_bolt_nut = _MultiFileDownloadableDatasetLoader(
    _bolt_nut_files_func,
    load_func=_load_as_multiblock,
)


@_deprecate_positional_args
def download_clown(load=True):  # noqa: FBT002
    """Download clown dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_clown()
    >>> dataset.plot()

    .. seealso::

        :ref:`Clown Dataset <clown_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_clown, load=load)


_dataset_clown = _SingleFileDownloadableDatasetLoader('clown.facet')


@_deprecate_positional_args
def download_topo_global(load=True):  # noqa: FBT002
    """Download topo dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_topo_global()
    >>> dataset.plot(cmap='gist_earth')

    .. seealso::

        :ref:`Topo Global Dataset <topo_global_dataset>`
            See this dataset in the Dataset Gallery for more info.

        This dataset is used in the following examples:

        * :ref:`compute_normals_example`
        * :ref:`background_image_example`

    """
    return _download_dataset(_dataset_topo_global, load=load)


_dataset_topo_global = _SingleFileDownloadableDatasetLoader('EarthModels/ETOPO_10min_Ice.vtp')


@_deprecate_positional_args
def download_topo_land(load=True):  # noqa: FBT002
    """Download topo land dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_topo_land()
    >>> dataset.plot(clim=[-2000, 3000], cmap='gist_earth', show_scalar_bar=False)

    .. seealso::

        :ref:`Topo Land Dataset <topo_land_dataset>`
            See this dataset in the Dataset Gallery for more info.

        This dataset is used in the following examples:

        * :ref:`geodesic_example`
        * :ref:`background_image_example`

    """
    return _download_dataset(_dataset_topo_land, load=load)


_dataset_topo_land = _SingleFileDownloadableDatasetLoader(
    'EarthModels/ETOPO_10min_Ice_only-land.vtp',
)


@_deprecate_positional_args
def download_coastlines(load=True):  # noqa: FBT002
    """Download coastlines dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_coastlines()
    >>> dataset.plot()

    .. seealso::

        :ref:`Coastlines Dataset <coastlines_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_coastlines, load=load)


_dataset_coastlines = _SingleFileDownloadableDatasetLoader('EarthModels/Coastlines_Los_Alamos.vtp')


@_deprecate_positional_args
def download_knee(load=True):  # noqa: FBT002
    """Download knee dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ImageData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_knee()
    >>> dataset.plot(cpos='xy', show_scalar_bar=False)

    .. seealso::

        :ref:`Knee Dataset <knee_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Knee Full Dataset <knee_full_dataset>`

        :ref:`medical_dataset_gallery`
            Browse other medical datasets.

        This dataset is used in the following examples:

        * :ref:`opacity_example`
        * :ref:`volume_rendering_example`
        * :ref:`slider_bar_widget_example`

    """
    return _download_dataset(_dataset_knee, load=load)


_dataset_knee = _SingleFileDownloadableDatasetLoader('DICOM_KNEE.dcm')


@_deprecate_positional_args
def download_knee_full(load=True):  # noqa: FBT002
    """Download full knee dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ImageData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_knee_full()
    >>> cpos = [
    ...     (-381.74, -46.02, 216.54),
    ...     (74.8305, 89.2905, 100.0),
    ...     (0.23, 0.072, 0.97),
    ... ]
    >>> dataset.plot(volume=True, cmap='bone', cpos=cpos, show_scalar_bar=False)

    .. seealso::

        :ref:`Knee Full Dataset <knee_full_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Knee Dataset <knee_dataset>`

        :ref:`medical_dataset_gallery`
            Browse other medical datasets.

        This dataset is used in the following examples:

        * :ref:`volume_rendering_example`
        * :ref:`slider_bar_widget_example`

    """
    return _download_dataset(_dataset_knee_full, load=load)


_dataset_knee_full = _SingleFileDownloadableDatasetLoader('vw_knee.slc')


@_deprecate_positional_args
def download_lidar(load=True):  # noqa: FBT002
    """Download lidar dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_lidar()
    >>> dataset.plot(cmap='gist_earth')

    .. seealso::

        :ref:`Lidar Dataset <lidar_dataset>`
            See this dataset in the Dataset Gallery for more info.

        This dataset is used in the following examples:

        * :ref:`create_point_cloud_example`
        * :ref:`edl_example`

    """
    return _download_dataset(_dataset_lidar, load=load)


_dataset_lidar = _SingleFileDownloadableDatasetLoader('kafadar-lidar-interp.vtp')


@_deprecate_positional_args
def download_exodus(load=True):  # noqa: FBT002
    """Sample ExodusII data file.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.MultiBlock | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_exodus()
    >>> dataset.plot()

    .. seealso::

        :ref:`Exodus Dataset <exodus_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_exodus, load=load)


_dataset_exodus = _SingleFileDownloadableDatasetLoader('mesh_fs8.exo')


@_deprecate_positional_args
def download_nefertiti(load=True):  # noqa: FBT002
    """Download mesh of Queen Nefertiti.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_nefertiti()
    >>> dataset.plot(cpos='xz')

    .. seealso::

        :ref:`Nefertiti Dataset <nefertiti_dataset>`
            See this dataset in the Dataset Gallery for more info.

        This dataset is used in the following examples:

        * :ref:`compute_normals_example`
        * :ref:`extract_edges_example`
        * :ref:`show_edges_example`
        * :ref:`edl_example`
        * :ref:`pbr_example`
        * :ref:`box_widget_example`

    """
    return _download_dataset(_dataset_nefertiti, load=load)


_dataset_nefertiti = _SingleFileDownloadableDatasetLoader(
    'nefertiti.ply.zip',
    target_file='nefertiti.ply',
)


@_deprecate_positional_args
def download_blood_vessels(load=True):  # noqa: FBT002
    """Download data representing the bifurcation of blood vessels.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_blood_vessels()
    >>> dataset.plot()

    .. seealso::

        :ref:`Blood Vessels Dataset <blood_vessels_dataset>`
            See this dataset in the Dataset Gallery for more info.

        This dataset is used in the following examples:

        * :ref:`read_parallel_example`
        * :ref:`streamlines_example`
        * :ref:`integrate_data_example`

    """
    return _download_dataset(_dataset_blood_vessels, load=load)


def _blood_vessels_load_func(obj):
    obj.set_active_vectors('velocity')
    return obj


_dataset_blood_vessels = _SingleFileDownloadableDatasetLoader(
    'pvtu_blood_vessels/blood_vessels.zip',
    target_file='T0000000500.pvtu',
    load_func=_blood_vessels_load_func,
)


@_deprecate_positional_args
def download_iron_protein(load=True):  # noqa: FBT002
    """Download iron protein dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ImageData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_iron_protein()
    >>> dataset.plot(volume=True, cmap='blues')

    .. seealso::

        :ref:`Iron Protein Dataset <iron_protein_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_iron_protein, load=load)


_dataset_iron_protein = _SingleFileDownloadableDatasetLoader('ironProt.vtk')


@_deprecate_positional_args
def download_tetrahedron(load=True):  # noqa: FBT002
    """Download tetrahedron dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Shrink and plot the dataset to show it is composed of several
    tetrahedrons.

    >>> from pyvista import examples
    >>> dataset = examples.download_tetrahedron()
    >>> dataset.shrink(0.85).plot()

    .. seealso::

        :ref:`Tetrahedron Dataset <tetrahedron_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_tetrahedron, load=load)


_dataset_tetrahedron = _SingleFileDownloadableDatasetLoader('Tetrahedron.vtu')


@_deprecate_positional_args
def download_saddle_surface(load=True):  # noqa: FBT002
    """Download saddle surface dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_saddle_surface()
    >>> dataset.plot()

    .. seealso::

        :ref:`Saddle Surface Dataset <saddle_surface_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`interpolate_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_saddle_surface, load=load)


_dataset_saddle_surface = _SingleFileDownloadableDatasetLoader('InterpolatingOnSTL_final.stl')


@_deprecate_positional_args
def download_sparse_points(load=True):  # noqa: FBT002
    """Download sparse points data.

    Used with :func:`download_saddle_surface`.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_sparse_points()
    >>> dataset.plot(scalars='val', render_points_as_spheres=True, point_size=50)

    .. seealso::

        :ref:`Sparse Points Dataset <sparse_points_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`interpolate_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_sparse_points, load=load)


def _sparse_points_reader(saved_file):
    points_reader = _vtk.vtkDelimitedTextReader()
    points_reader.SetFileName(saved_file)
    points_reader.DetectNumericColumnsOn()
    points_reader.SetFieldDelimiterCharacters('\t')
    points_reader.SetHaveHeaders(True)
    table_points = _vtk.vtkTableToPolyData()
    table_points.SetInputConnection(points_reader.GetOutputPort())
    table_points.SetXColumn('x')
    table_points.SetYColumn('y')
    table_points.SetZColumn('z')
    table_points.Update()
    return pyvista.wrap(table_points.GetOutput())


_dataset_sparse_points = _SingleFileDownloadableDatasetLoader(
    'sparsePoints.txt',
    read_func=_sparse_points_reader,
)


@_deprecate_positional_args
def download_foot_bones(load=True):  # noqa: FBT002
    """Download foot bones dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_foot_bones()
    >>> dataset.plot()

    .. seealso::

        :ref:`Foot Bones Dataset <foot_bones_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`voxelize_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_foot_bones, load=load)


_dataset_foot_bones = _SingleFileDownloadableDatasetLoader('fsu/footbones.ply')


@_deprecate_positional_args
def download_guitar(load=True):  # noqa: FBT002
    """Download guitar dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_guitar()
    >>> dataset.plot()

    .. seealso::

        :ref:`Guitar Dataset <guitar_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Trumpet Dataset <trumpet_dataset>`

    """
    return _download_dataset(_dataset_guitar, load=load)


_dataset_guitar = _SingleFileDownloadableDatasetLoader('fsu/stratocaster.ply')


@_deprecate_positional_args
def download_quadratic_pyramid(load=True):  # noqa: FBT002
    """Download quadratic pyramid dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Shrink and plot the dataset to show it is composed of several
    pyramids.

    >>> from pyvista import examples
    >>> dataset = examples.download_quadratic_pyramid()
    >>> dataset.shrink(0.4).plot()

    .. seealso::

        :ref:`Quadratic Pyramid Dataset <quadratic_pyramid_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_quadratic_pyramid, load=load)


_dataset_quadratic_pyramid = _SingleFileDownloadableDatasetLoader('QuadraticPyramid.vtu')


@_deprecate_positional_args
def download_bird(load=True):  # noqa: FBT002
    """Download bird dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ImageData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_bird()
    >>> dataset.plot(rgba=True, cpos='xy')

    .. seealso::

        :ref:`Bird Dataset <bird_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Bird Texture Dataset <bird_texture_dataset>`

    """
    return _download_dataset(_dataset_bird, load=load)


_dataset_bird = _SingleFileDownloadableDatasetLoader('Pileated.jpg')


@_deprecate_positional_args
def download_bird_texture(load=True):  # noqa: FBT002
    """Download bird texture.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.Texture | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_bird_texture()
    >>> dataset.plot(cpos='xy')

    .. seealso::

        :ref:`Bird Texture Dataset <bird_texture_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Bird Dataset <bird_dataset>`

    """
    return _download_dataset(_dataset_bird_texture, load=load)


_dataset_bird_texture = _SingleFileDownloadableDatasetLoader(
    'Pileated.jpg',
    read_func=read_texture,  # type: ignore[arg-type]
)


@_deprecate_positional_args
def download_office(load=True):  # noqa: FBT002
    """Download office dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.StructuredGrid | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_office()
    >>> dataset.contour().plot()

    .. seealso::

        :ref:`Office Dataset <office_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`clip_with_plane_box_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_office, load=load)


_dataset_office = _SingleFileDownloadableDatasetLoader('office.binary.vtk')


@_deprecate_positional_args
def download_horse_points(load=True):  # noqa: FBT002
    """Download horse points dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_horse_points()
    >>> dataset.plot(point_size=1)

    .. seealso::

        :ref:`Horse Points Dataset <horse_points_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Horse Dataset <horse_dataset>`

    """
    return _download_dataset(_dataset_horse_points, load=load)


_dataset_horse_points = _SingleFileDownloadableDatasetLoader('horsePoints.vtp')


@_deprecate_positional_args
def download_horse(load=True):  # noqa: FBT002
    """Download horse dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.DataSet | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_horse()
    >>> dataset.plot(smooth_shading=True)

    .. seealso::

        :ref:`Horse Dataset <horse_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Horse Points Dataset <horse_points_dataset>`

        :ref:`mesh_lighting_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_horse, load=load)


_dataset_horse = _SingleFileDownloadableDatasetLoader('horse.vtp')


@_deprecate_positional_args
def download_cake_easy(load=True):  # noqa: FBT002
    """Download cake dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ImageData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_cake_easy()
    >>> dataset.plot(rgba=True, cpos='xy')

    .. seealso::

        :ref:`Cake Easy Dataset <cake_easy_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Cake Easy Texture Dataset <cake_easy_texture_dataset>`

    """
    return _download_dataset(_dataset_cake_easy, load=load)


_dataset_cake_easy = _SingleFileDownloadableDatasetLoader('cake_easy.jpg')


@_deprecate_positional_args
def download_cake_easy_texture(load=True):  # noqa: FBT002
    """Download cake texture.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.Texture | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_cake_easy_texture()
    >>> dataset.plot(cpos='xy')

    .. seealso::

        :ref:`Cake Easy Texture Dataset <cake_easy_texture_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Cake Easy Dataset <cake_easy_dataset>`

    """
    return _download_dataset(_dataset_cake_easy_texture, load=load)


_dataset_cake_easy_texture = _SingleFileDownloadableDatasetLoader(
    'cake_easy.jpg',
    read_func=read_texture,  # type: ignore[arg-type]
)


@_deprecate_positional_args
def download_rectilinear_grid(load=True):  # noqa: FBT002
    """Download rectilinear grid dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.RectilinearGrid | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Compute the threshold of this dataset.

    >>> from pyvista import examples
    >>> dataset = examples.download_rectilinear_grid()
    >>> dataset.threshold(0.0001).plot()

    .. seealso::

        :ref:`Rectilinear Grid Dataset <rectilinear_grid_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_rectilinear_grid, load=load)


_dataset_rectilinear_grid = _SingleFileDownloadableDatasetLoader('RectilinearGrid.vtr')


@_deprecate_positional_args
def download_gourds(zoom=False, load=True):  # noqa: FBT002
    """Download gourds dataset.

    Parameters
    ----------
    zoom : bool, default: False
        When ``True``, return the zoomed picture of the gourds.

    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ImageData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_gourds()
    >>> dataset.plot(rgba=True, cpos='xy')

    .. seealso::

        :ref:`Gourds Dataset <gourds_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Gourds Pnm Dataset <gourds_pnm_dataset>`

        :ref:`Gourds Texture Dataset <gourds_texture_dataset>`

        :ref:`gaussian_smoothing_example`
            Example using this dataset.

    """
    example = __gourds2 if zoom else _dataset_gourds
    return _download_dataset(example, load=load)


# Two loadable files, but only one example
# Name variables such that non-zoomed version is the 'representative' example
# Use '__' on the zoomed version to label it as private
_dataset_gourds = _SingleFileDownloadableDatasetLoader('Gourds.png')
__gourds2 = _SingleFileDownloadableDatasetLoader('Gourds2.jpg')


@_deprecate_positional_args
def download_gourds_texture(zoom=False, load=True):  # noqa: FBT002
    """Download gourds texture.

    Parameters
    ----------
    zoom : bool, default: False
        When ``True``, return the zoomed picture of the gourds.

    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.DataSet | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_gourds_texture()
    >>> dataset.plot(cpos='xy')

    .. seealso::

        :ref:`Gourds Texture Dataset <gourds_texture_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Gourds Dataset <gourds_dataset>`

        :ref:`Gourds Pnm Dataset <gourds_pnm_dataset>`

    """
    example = __gourds2_texture if zoom else _dataset_gourds_texture
    return _download_dataset(example, load=load)


# Two loadable files, but only one example
# Name variables such that non-zoomed version is the 'representative' example
# Use '__' on the zoomed version to label it as private
_dataset_gourds_texture = _SingleFileDownloadableDatasetLoader(
    'Gourds.png',
    read_func=read_texture,  # type: ignore[arg-type]
)
__gourds2_texture = _SingleFileDownloadableDatasetLoader('Gourds2.jpg', read_func=read_texture)  # type: ignore[arg-type]


@_deprecate_positional_args
def download_gourds_pnm(load=True):  # noqa: FBT002
    """Download gourds dataset from pnm file.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ImageData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_gourds_pnm()
    >>> dataset.plot(rgba=True, cpos='xy')

    .. seealso::

        :ref:`Gourds Pnm Dataset <gourds_pnm_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Gourds Dataset <gourds_dataset>`

        :ref:`Gourds Texture Dataset <gourds_texture_dataset>`

    """
    return _download_dataset(_dataset_gourds_pnm, load=load)


_dataset_gourds_pnm = _SingleFileDownloadableDatasetLoader('Gourds.pnm')


@_deprecate_positional_args
def download_unstructured_grid(load=True):  # noqa: FBT002
    """Download unstructured grid dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_unstructured_grid()
    >>> dataset.plot(show_edges=True)

    .. seealso::

        :ref:`Unstructured Grid Dataset <unstructured_grid_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_unstructured_grid, load=load)


_dataset_unstructured_grid = _SingleFileDownloadableDatasetLoader('uGridEx.vtk')


@_deprecate_positional_args
def download_letter_k(load=True):  # noqa: FBT002
    """Download letter k dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_letter_k()
    >>> dataset.plot(cpos='xy')

    .. seealso::

        :ref:`Letter K Dataset <letter_k_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Letter A Dataset <letter_a_dataset>`

    """
    return _download_dataset(_dataset_letter_k, load=load)


_dataset_letter_k = _SingleFileDownloadableDatasetLoader('k.vtk')


@_deprecate_positional_args
def download_letter_a(load=True):  # noqa: FBT002
    """Download letter a dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_letter_a()
    >>> dataset.plot(cpos='xy', show_edges=True)

    .. seealso::

        :ref:`Letter A Dataset <letter_a_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Letter K Dataset <letter_k_dataset>`

        :ref:`cell_centers_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_letter_a, load=load)


_dataset_letter_a = _SingleFileDownloadableDatasetLoader('a_grid.vtk')


@_deprecate_positional_args
def download_poly_line(load=True):  # noqa: FBT002
    """Download polyline dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_poly_line()
    >>> dataset.plot(line_width=5)

    .. seealso::

        :ref:`Poly Line Dataset <poly_line_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_poly_line, load=load)


_dataset_poly_line = _SingleFileDownloadableDatasetLoader('polyline.vtk')


@_deprecate_positional_args
def download_cad_model(load=True):  # noqa: FBT002
    """Download cad dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_cad_model()
    >>> dataset.plot()

    .. seealso::

        :ref:`Cad Model Dataset <cad_model_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`read_file_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_cad_model, load=load)


_dataset_cad_model = _SingleFileDownloadableDatasetLoader('42400-IDGH.stl')


@_deprecate_positional_args
def download_frog(load=True):  # noqa: FBT002
    """Download frog dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ImageData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> cpos = [
    ...     [8.4287e02, -5.7418e02, -4.4085e02],
    ...     [2.4950e02, 2.3450e02, 1.0125e02],
    ...     [-3.2000e-01, 3.5000e-01, -8.8000e-01],
    ... ]
    >>> dataset = examples.download_frog()
    >>> dataset.plot(volume=True, cpos=cpos)

    .. seealso::

        :ref:`Frog Dataset <frog_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Frog Tissues Dataset <frog_tissues_dataset>`
            Segmentation labels associated with this dataset.

        :ref:`medical_dataset_gallery`
            Browse other medical datasets.

        :ref:`volume_rendering_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_frog, load=load)


def _frog_files_func():
    # Multiple files needed for read, but only one gets loaded
    frog_zraw = _DownloadableFile('froggy/frog.zraw')
    frog_mhd = _SingleFileDownloadableDatasetLoader('froggy/frog.mhd')
    return frog_mhd, frog_zraw


_dataset_frog = _MultiFileDownloadableDatasetLoader(_frog_files_func)


@_deprecate_positional_args
def download_chest(load=True):  # noqa: FBT002
    """Download chest dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ImageData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_chest()
    >>> dataset.plot(cpos='xy')

    .. seealso::

        :ref:`Chest Dataset <chest_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`medical_dataset_gallery`
            Browse other medical datasets.

        :ref:`volume_rendering_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_chest, load=load)


_dataset_chest = _SingleFileDownloadableDatasetLoader('MetaIO/ChestCT-SHORT.mha')


@_deprecate_positional_args
def download_brain_atlas_with_sides(load=True):  # noqa: FBT002
    """Download an image of an averaged brain with a right-left label.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ImageData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_brain_atlas_with_sides()
    >>> dataset.slice(normal='z').plot(cpos='xy')

    .. seealso::

        :ref:`Brain Atlas With Sides Dataset <brain_atlas_with_sides_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Brain Dataset <brain_dataset>`

        :ref:`medical_dataset_gallery`
            Browse other medical datasets.

    """
    return _download_dataset(_dataset_brain_atlas_with_sides, load=load)


_dataset_brain_atlas_with_sides = _SingleFileDownloadableDatasetLoader('avg152T1_RL_nifti.nii.gz')


@_deprecate_positional_args
def download_prostate(load=True):  # noqa: FBT002
    """Download prostate dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ImageData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_prostate()
    >>> dataset.plot(cpos='xy')

    .. seealso::

        :ref:`Prostate Dataset <prostate_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`medical_dataset_gallery`
            Browse other medical datasets.

    """
    return _download_dataset(_dataset_prostate, load=load)


_dataset_prostate = _SingleFileDownloadableDatasetLoader('prostate.img')


@_deprecate_positional_args
def download_filled_contours(load=True):  # noqa: FBT002
    """Download filled contours dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_filled_contours()
    >>> dataset.plot(cpos='xy')

    .. seealso::

        :ref:`Filled Contours Dataset <filled_contours_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_filled_contours, load=load)


_dataset_filled_contours = _SingleFileDownloadableDatasetLoader('filledContours.vtp')


@_deprecate_positional_args
def download_doorman(load=True):  # noqa: FBT002
    """Download doorman dataset.

    .. versionchanged:: 0.44.0
        Add support for downloading the texture images.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_doorman()
    >>> dataset.plot(cpos='xy')

    .. seealso::

        :ref:`Doorman Dataset <doorman_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`read_file_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_doorman, load=load)


def _doorman_files_func():
    # Multiple files needed for read, but only one gets loaded
    doorman_obj = _SingleFileDownloadableDatasetLoader('doorman/doorman.obj')
    doorman_mtl = _DownloadableFile('doorman/doorman.mtl')
    t_doorMan_d = _DownloadableFile('doorman/t_doorMan_d.png')
    t_doorMan_n = _DownloadableFile('doorman/t_doorMan_n.png')
    t_doorMan_s = _DownloadableFile('doorman/t_doorMan_s.png')
    t_doorMan_teeth_d = _DownloadableFile('doorman/t_doorMan_teeth_d.png')
    t_doorMan_teeth_n = _DownloadableFile('doorman/t_doorMan_teeth_n.png')
    t_eye_d = _DownloadableFile('doorman/t_eye_d.png')
    t_eye_n = _DownloadableFile('doorman/t_eye_n.png')
    return (
        doorman_obj,
        doorman_mtl,
        t_doorMan_d,
        t_doorMan_n,
        t_doorMan_s,
        t_doorMan_teeth_d,
        t_doorMan_teeth_n,
        t_eye_d,
        t_eye_n,
    )


_dataset_doorman = _MultiFileDownloadableDatasetLoader(
    files_func=_doorman_files_func,
)


@_deprecate_positional_args
def download_mug(load=True):  # noqa: FBT002
    """Download mug dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.MultiBlock | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_mug()
    >>> dataset.plot()

    .. seealso::

        :ref:`Mug Dataset <mug_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_mug, load=load)


_dataset_mug = _SingleFileDownloadableDatasetLoader('mug.e')


@_deprecate_positional_args
def download_oblique_cone(load=True):  # noqa: FBT002
    """Download oblique cone dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_oblique_cone()
    >>> dataset.plot()

    .. seealso::

        :ref:`Oblique Cone Dataset <oblique_cone_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_oblique_cone, load=load)


_dataset_oblique_cone = _SingleFileDownloadableDatasetLoader('ObliqueCone.vtp')


@_deprecate_positional_args
def download_emoji(load=True):  # noqa: FBT002
    """Download emoji dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ImageData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_emoji()
    >>> dataset.plot(rgba=True, cpos='xy')

    .. seealso::

        :ref:`Emoji Dataset <emoji_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Emoji Texture Dataset <emoji_texture_dataset>`

    """
    return _download_dataset(_dataset_emoji, load=load)


_dataset_emoji = _SingleFileDownloadableDatasetLoader('emote.jpg')


@_deprecate_positional_args
def download_emoji_texture(load=True):  # noqa: FBT002
    """Download emoji texture.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.Texture | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_emoji_texture()
    >>> dataset.plot(cpos='xy')

    .. seealso::

        :ref:`Emoji Texture Dataset <emoji_texture_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Emoji Dataset <emoji_dataset>`

    """
    return _download_dataset(_dataset_emoji_texture, load=load)


_dataset_emoji_texture = _SingleFileDownloadableDatasetLoader('emote.jpg', read_func=read_texture)  # type: ignore[arg-type]


@_deprecate_positional_args
def download_teapot(load=True):  # noqa: FBT002
    """Download teapot dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_teapot()
    >>> dataset.plot(cpos='xy')

    .. seealso::

        :ref:`Teapot Dataset <teapot_dataset>`
            See this dataset in the Dataset Gallery for more info.

        This dataset is used in the following examples:

        * :ref:`read_file_example`
        * :ref:`cell_centers_example`

    """
    return _download_dataset(_dataset_teapot, load=load)


_dataset_teapot = _SingleFileDownloadableDatasetLoader('teapot.g')


@_deprecate_positional_args
def download_brain(load=True):  # noqa: FBT002
    """Download brain dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ImageData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_brain()
    >>> dataset.plot(volume=True)

    .. seealso::

        :ref:`Brain Dataset <brain_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Brain Atlas With Sides Dataset <brain_atlas_with_sides_dataset>`

        :ref:`medical_dataset_gallery`
            Browse other medical datasets.

        This dataset is used in the following examples:

        * :ref:`gaussian_smoothing_example`
        * :ref:`slice_example`
        * :ref:`depth_peeling_example`
        * :ref:`moving_isovalue_example`
        * :ref:`plane_widget_example`

    """
    return _download_dataset(_dataset_brain, load=load)


_dataset_brain = _SingleFileDownloadableDatasetLoader('brain.vtk')


@_deprecate_positional_args
def download_structured_grid(load=True):  # noqa: FBT002
    """Download structured grid dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.StructuredGrid | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_structured_grid()
    >>> dataset.plot(show_edges=True)

    .. seealso::

        :ref:`Structured Grid Dataset <structured_grid_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Structured Grid Two Dataset <structured_grid_two_dataset>`

    """
    return _download_dataset(_dataset_structured_grid, load=load)


_dataset_structured_grid = _SingleFileDownloadableDatasetLoader('StructuredGrid.vts')


@_deprecate_positional_args
def download_structured_grid_two(load=True):  # noqa: FBT002
    """Download structured grid two dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.StructuredGrid | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_structured_grid_two()
    >>> dataset.plot(show_edges=True)

    .. seealso::

        :ref:`Structured Grid Two Dataset <structured_grid_two_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Structured Grid Dataset <structured_grid_dataset>`

    """
    return _download_dataset(_dataset_structured_grid_two, load=load)


_dataset_structured_grid_two = _SingleFileDownloadableDatasetLoader('SampleStructGrid.vtk')


@_deprecate_positional_args
def download_trumpet(load=True):  # noqa: FBT002
    """Download trumpet dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_trumpet()
    >>> dataset.plot()

    .. seealso::

        :ref:`Trumpet Dataset <trumpet_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Guitar Dataset <guitar_dataset>`

    """
    return _download_dataset(_dataset_trumpet, load=load)


_dataset_trumpet = _SingleFileDownloadableDatasetLoader('trumpet.obj')


@_deprecate_positional_args
def download_face(load=True):  # noqa: FBT002
    """Download face dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_face()
    >>> dataset.plot()

    .. seealso::

        :ref:`Face Dataset <face_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Face2 Dataset <face2_dataset>`

        :ref:`decimate_example`
            Example using this dataset.

    """
    # TODO: there is a texture with this
    return _download_dataset(_dataset_face, load=load)


_dataset_face = _SingleFileDownloadableDatasetLoader('fran_cut.vtk')


@_deprecate_positional_args
def download_sky_box_nz(load=True):  # noqa: FBT002
    """Download skybox-nz dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ImageData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_sky_box_nz()
    >>> dataset.plot(rgba=True, cpos='xy')

    .. seealso::

        :ref:`Sky Box Nz Dataset <sky_box_nz_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Sky Box Nz Texture Dataset <sky_box_nz_texture_dataset>`

        :ref:`Sky Box Cube Map Dataset <sky_box_cube_map_dataset>`

    """
    return _download_dataset(_dataset_sky_box_nz, load=load)


_dataset_sky_box_nz = _SingleFileDownloadableDatasetLoader('skybox-nz.jpg')


@_deprecate_positional_args
def download_sky_box_nz_texture(load=True):  # noqa: FBT002
    """Download skybox-nz texture.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.Texture | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_sky_box_nz_texture()
    >>> dataset.plot(cpos='xy')

    .. seealso::

        :ref:`Sky Box Nz Texture Dataset <sky_box_nz_texture_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Sky Box Nz Dataset <sky_box_nz_dataset>`

        :ref:`Sky Box Cube Map Dataset <sky_box_cube_map_dataset>`

    """
    return _download_dataset(_dataset_sky_box_nz_texture, load=load)


_dataset_sky_box_nz_texture = _SingleFileDownloadableDatasetLoader(
    'skybox-nz.jpg',
    read_func=read_texture,  # type: ignore[arg-type]
)


@_deprecate_positional_args
def download_disc_quads(load=True):  # noqa: FBT002
    """Download disc quads dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_disc_quads()
    >>> dataset.plot(show_edges=True)

    .. seealso::

        :ref:`Disc Quads Dataset <disc_quads_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_disc_quads, load=load)


_dataset_disc_quads = _SingleFileDownloadableDatasetLoader('Disc_BiQuadraticQuads_0_0.vtu')


@_deprecate_positional_args
def download_honolulu(load=True):  # noqa: FBT002
    """Download honolulu dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_honolulu()
    >>> dataset.plot(
    ...     scalars=dataset.points[:, 2],
    ...     show_scalar_bar=False,
    ...     cmap='gist_earth',
    ...     clim=[-50, 800],
    ... )

    .. seealso::

        :ref:`Honolulu Dataset <honolulu_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_honolulu, load=load)


_dataset_honolulu = _SingleFileDownloadableDatasetLoader('honolulu.vtk')


@_deprecate_positional_args
def download_motor(load=True):  # noqa: FBT002
    """Download motor dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_motor()
    >>> dataset.plot()

    .. seealso::

        :ref:`Motor Dataset <motor_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_motor, load=load)


_dataset_motor = _SingleFileDownloadableDatasetLoader('motor.g')


@_deprecate_positional_args
def download_tri_quadratic_hexahedron(load=True):  # noqa: FBT002
    """Download tri quadratic hexahedron dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_tri_quadratic_hexahedron()
    >>> dataset.plot()

    Show non-linear subdivision.

    >>> surf = dataset.extract_surface(nonlinear_subdivision=5)
    >>> surf.plot(smooth_shading=True)

    .. seealso::

        :ref:`Tri Quadratic Hexahedron Dataset <tri_quadratic_hexahedron_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_tri_quadratic_hexahedron, load=load)


def _tri_quadratic_hexahedron_load_func(dataset):
    dataset.clear_data()
    return dataset


_dataset_tri_quadratic_hexahedron = _SingleFileDownloadableDatasetLoader(
    'TriQuadraticHexahedron.vtu',
    load_func=_tri_quadratic_hexahedron_load_func,
)


@_deprecate_positional_args
def download_human(load=True):  # noqa: FBT002
    """Download human dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_human()
    >>> dataset.plot(scalars='Color', rgb=True)

    .. seealso::

        :ref:`Human Dataset <human_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_human, load=load)


_dataset_human = _SingleFileDownloadableDatasetLoader('Human.vtp')


@_deprecate_positional_args
def download_vtk(load=True):  # noqa: FBT002
    """Download vtk dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_vtk()
    >>> dataset.plot(cpos='xy', line_width=5)

    .. seealso::

        :ref:`Vtk Dataset <vtk_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Vtk Logo Dataset <vtk_logo_dataset>`

    """
    return _download_dataset(_dataset_vtk, load=load)


_dataset_vtk = _SingleFileDownloadableDatasetLoader('vtk.vtp')


@_deprecate_positional_args
def download_spider(load=True):  # noqa: FBT002
    """Download spider dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_spider()
    >>> dataset.plot()

    .. seealso::

        :ref:`Spider Dataset <spider_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_spider, load=load)


_dataset_spider = _SingleFileDownloadableDatasetLoader('spider.ply')


@_deprecate_positional_args
def download_carotid(load=True):  # noqa: FBT002
    """Download carotid dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ImageData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> cpos = [
    ...     [220.96, -24.38, -69.96],
    ...     [135.86, 106.55, 17.72],
    ...     [-0.25, 0.42, -0.87],
    ... ]
    >>> dataset = examples.download_carotid()
    >>> dataset.plot(volume=True, cpos=cpos)

    .. seealso::

        :ref:`Carotid Dataset <carotid_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`medical_dataset_gallery`
            Browse other medical datasets.

        This dataset is used in the following examples:

        * :ref:`glyph_example`
        * :ref:`gradients_example`
        * :ref:`streamlines_example`
        * :ref:`plane_widget_example`

    """
    return _download_dataset(_dataset_carotid, load=load)


def _carotid_load_func(mesh):
    mesh.set_active_scalars('scalars')
    mesh.set_active_vectors('vectors')
    return mesh


_dataset_carotid = _SingleFileDownloadableDatasetLoader(
    'carotid.vtk', load_func=_carotid_load_func
)


@_deprecate_positional_args
def download_blow(load=True):  # noqa: FBT002
    """Download blow dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> cpos = [
    ...     [71.96, 86.1, 28.45],
    ...     [3.5, 12.0, 1.0],
    ...     [-0.18, -0.19, 0.96],
    ... ]
    >>> dataset = examples.download_blow()
    >>> dataset.plot(
    ...     scalars='displacement1',
    ...     component=1,
    ...     cpos=cpos,
    ...     show_scalar_bar=False,
    ...     smooth_shading=True,
    ... )

    .. seealso::

        :ref:`Blow Dataset <blow_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_blow, load=load)


_dataset_blow = _SingleFileDownloadableDatasetLoader('blow.vtk')


@_deprecate_positional_args
def download_shark(load=True):  # noqa: FBT002
    """Download shark dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> cpos = [
    ...     [-2.3195e02, -3.3930e01, 1.2981e02],
    ...     [-8.7100e00, 1.9000e-01, -1.1740e01],
    ...     [-1.4000e-01, 9.9000e-01, 2.0000e-02],
    ... ]
    >>> dataset = examples.download_shark()
    >>> dataset.plot(cpos=cpos, smooth_shading=True)

    .. seealso::

        :ref:`Shark Dataset <shark_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Great White Shark Dataset <great_white_shark_dataset>`
            Similar dataset.

        :ref:`Grey Nurse Shark Dataset <grey_nurse_shark_dataset>`
            Similar dataset.

    """
    return _download_dataset(_dataset_shark, load=load)


_dataset_shark = _SingleFileDownloadableDatasetLoader('shark.ply')


@_deprecate_positional_args
def download_great_white_shark(load=True):  # noqa: FBT002
    """Download great white shark dataset.

    .. versionadded:: 0.45

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> cpos = [(9.0, 1.0, 21.0), (-1.0, 2.0, -2.0), (0.0, 1.0, 0.0)]
    >>> dataset = examples.download_great_white_shark()
    >>> dataset.plot(cpos=cpos, smooth_shading=True)

    .. seealso::

        :ref:`Great White Shark Dataset <great_white_shark_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Shark Dataset <shark_dataset>`
            Similar dataset.

        :ref:`Grey Nurse Shark Dataset <grey_nurse_shark_dataset>`
            Similar dataset.

    """
    return _download_dataset(_dataset_great_white_shark, load=load)


_dataset_great_white_shark = _SingleFileDownloadableDatasetLoader(
    'great_white_shark/greatWhite.stl'
)


@_deprecate_positional_args
def download_grey_nurse_shark(load=True):  # noqa: FBT002
    """Download grey nurse shark dataset.

    .. versionadded:: 0.45

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> cpos = [
    ...     [-200, -100, -16.0],
    ...     [-20.0, 20.0, -2.00],
    ...     [0.00, 0.00, 1.00],
    ... ]
    >>> dataset = examples.download_grey_nurse_shark()
    >>> dataset.plot(cpos=cpos, smooth_shading=True)

    .. seealso::

        :ref:`Grey Nurse Shark Dataset <grey_nurse_shark_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Shark Dataset <shark_dataset>`
            Similar dataset.

        :ref:`Great White Shark Dataset <great_white_shark_dataset>`
            Similar dataset.

    """
    return _download_dataset(_dataset_grey_nurse_shark, load=load)


_dataset_grey_nurse_shark = _SingleFileDownloadableDatasetLoader(
    'grey_nurse_shark/Grey_Nurse_Shark.stl'
)


@_deprecate_positional_args
def download_dragon(load=True):  # noqa: FBT002
    """Download dragon dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_dragon()
    >>> dataset.plot(cpos='xy')

    .. seealso::

        :ref:`Dragon Dataset <dragon_dataset>`
            See this dataset in the Dataset Gallery for more info.

        This dataset is used in the following examples:

        * :ref:`floors_example`
        * :ref:`orbit_example`
        * :ref:`silhouette_example`
        * :ref:`shadows_example`

    """
    return _download_dataset(_dataset_dragon, load=load)


_dataset_dragon = _SingleFileDownloadableDatasetLoader('dragon.ply')


@_deprecate_positional_args
def download_armadillo(load=True):  # noqa: FBT002
    """Download armadillo dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Plot the armadillo dataset. Use a custom camera position.

    >>> from pyvista import examples
    >>> cpos = [
    ...     (161.5, 82.1, -330.2),
    ...     (-4.3, 24.5, -1.6),
    ...     (-0.1, 1, 0.12),
    ... ]
    >>> dataset = examples.download_armadillo()
    >>> dataset.plot(cpos=cpos)

    .. seealso::

        :ref:`Armadillo Dataset <armadillo_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_armadillo, load=load)


_dataset_armadillo = _SingleFileDownloadableDatasetLoader('Armadillo.ply')


@_deprecate_positional_args
def download_gears(load=True):  # noqa: FBT002
    """Download gears dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Download the dataset, split the bodies, and color each one.

    >>> import numpy as np
    >>> from pyvista import examples
    >>> dataset = examples.download_gears()
    >>> bodies = dataset.split_bodies()
    >>> for i, body in enumerate(bodies):
    ...     bid = np.empty(body.n_points)
    ...     bid[:] = i
    ...     body.point_data['Body ID'] = bid
    >>> bodies.plot(cmap='jet')

    .. seealso::

        :ref:`Gears Dataset <gears_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_gears, load=load)


_dataset_gears = _SingleFileDownloadableDatasetLoader('gears.stl')


@_deprecate_positional_args
def download_torso(load=True):  # noqa: FBT002
    """Download torso dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_torso()
    >>> dataset.plot(cpos='xz')

    .. seealso::

        :ref:`Torso Dataset <torso_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_torso, load=load)


_dataset_torso = _SingleFileDownloadableDatasetLoader('Torso.vtp')


@_deprecate_positional_args
def download_kitchen(split=False, load=True):  # noqa: FBT002
    """Download structured grid of kitchen with velocity field.

    Use the ``split`` argument to extract all of the furniture in the
    kitchen.

    Parameters
    ----------
    split : bool, default: False
        Optionally split the furniture and return a
        :class:`pyvista.MultiBlock`.

    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.StructuredGrid | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> dataset = examples.download_kitchen()
    >>> point_a = (0.08, 2.50, 0.71)
    >>> point_b = (0.08, 4.50, 0.71)
    >>> line = pv.Line(point_a, point_b, resolution=39)
    >>> streamlines = dataset.streamlines_from_source(line, max_length=200)
    >>> streamlines.plot(show_grid=True)

    .. seealso::

        :ref:`Kitchen Dataset <kitchen_dataset>`
            See this dataset in the Dataset Gallery for more info.

        This dataset is used in the following examples:

        * :ref:`plot_over_line_example`
        * :ref:`line_widget_example`

    """
    if load and split:
        return _download_dataset(__kitchen_split, load=load)
    else:
        return _download_dataset(_dataset_kitchen, load=load)


def _kitchen_split_load_func(mesh):
    extents = {
        'door': (27, 27, 14, 18, 0, 11),
        'window1': (0, 0, 9, 18, 6, 12),
        'window2': (5, 12, 23, 23, 6, 12),
        'klower1': (17, 17, 0, 11, 0, 6),
        'klower2': (19, 19, 0, 11, 0, 6),
        'klower3': (17, 19, 0, 0, 0, 6),
        'klower4': (17, 19, 11, 11, 0, 6),
        'klower5': (17, 19, 0, 11, 0, 0),
        'klower6': (17, 19, 0, 7, 6, 6),
        'klower7': (17, 19, 9, 11, 6, 6),
        'hood1': (17, 17, 0, 11, 11, 16),
        'hood2': (19, 19, 0, 11, 11, 16),
        'hood3': (17, 19, 0, 0, 11, 16),
        'hood4': (17, 19, 11, 11, 11, 16),
        'hood5': (17, 19, 0, 11, 16, 16),
        'cookingPlate': (17, 19, 7, 9, 6, 6),
        'furniture': (17, 19, 7, 9, 11, 11),
    }
    kitchen = pyvista.MultiBlock()
    for key, extent in extents.items():
        alg = _vtk.vtkStructuredGridGeometryFilter()
        alg.SetInputDataObject(mesh)
        alg.SetExtent(extent)  # type: ignore[call-overload]
        alg.Update()
        result = pyvista.core.filters._get_output(alg)
        kitchen[key] = result
    return kitchen


_dataset_kitchen = _SingleFileDownloadableDatasetLoader('kitchen.vtk')
__kitchen_split = _SingleFileDownloadableDatasetLoader(
    'kitchen.vtk',
    load_func=_kitchen_split_load_func,
)


@_deprecate_positional_args
def download_tetra_dc_mesh(load=True):  # noqa: FBT002
    """Download two meshes defining an electrical inverse problem.

    This contains a high resolution forward modeled mesh and a coarse
    inverse modeled mesh.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.MultiBlock
        DataSet containing the high resolution forward modeled mesh
        and a coarse inverse modeled mesh.

    Examples
    --------
    >>> from pyvista import examples
    >>> fine, coarse = examples.download_tetra_dc_mesh()
    >>> coarse.plot()

    .. seealso::

        :ref:`Tetra Dc Mesh Dataset <tetra_dc_mesh_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_tetra_dc_mesh, load=load)


def _tetra_dc_mesh_files_func():
    def _fwd_load_func(mesh):
        mesh.set_active_scalars('Resistivity(log10)-fwd')
        return mesh

    def _inv_load_func(mesh):
        mesh.set_active_scalars('Resistivity(log10)')
        return mesh

    fwd = _SingleFileDownloadableDatasetLoader(
        'dc-inversion.zip',
        target_file='mesh-forward.vtu',
        load_func=_fwd_load_func,
    )
    inv = _SingleFileDownloadableDatasetLoader(
        'dc-inversion.zip',
        target_file='mesh-inverse.vtu',
        load_func=_inv_load_func,
    )
    return fwd, inv


_dataset_tetra_dc_mesh = _MultiFileDownloadableDatasetLoader(
    _tetra_dc_mesh_files_func,
    load_func=functools.partial(_load_as_multiblock, names=['forward', 'inverse']),
)


@_deprecate_positional_args
def download_model_with_variance(load=True):  # noqa: FBT002
    """Download model with variance dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_model_with_variance()
    >>> dataset.plot()

    .. seealso::

        :ref:`Model With Variance Dataset <model_with_variance_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`opacity_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_model_with_variance, load=load)


_dataset_model_with_variance = _SingleFileDownloadableDatasetLoader('model_with_variance.vtu')


@_deprecate_positional_args
def download_thermal_probes(load=True):  # noqa: FBT002
    """Download thermal probes dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_thermal_probes()
    >>> dataset.plot(render_points_as_spheres=True, point_size=5, cpos='xy')

    .. seealso::

        :ref:`Thermal Probes Dataset <thermal_probes_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`interpolate_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_thermal_probes, load=load)


_dataset_thermal_probes = _SingleFileDownloadableDatasetLoader('probes.vtp')


@_deprecate_positional_args
def download_carburetor(load=True):  # noqa: FBT002
    """Download scan of a carburetor.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_carburetor()
    >>> dataset.plot()

    .. seealso::

        :ref:`Carburetor Dataset <carburetor_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_carburetor, load=load)


_dataset_carburetor = _SingleFileDownloadableDatasetLoader('carburetor.ply')


@_deprecate_positional_args
def download_turbine_blade(load=True):  # noqa: FBT002
    """Download scan of a turbine blade.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_turbine_blade()
    >>> dataset.plot()

    .. seealso::

        :ref:`Turbine Blade Dataset <turbine_blade_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_turbine_blade, load=load)


_dataset_turbine_blade = _SingleFileDownloadableDatasetLoader('turbineblade.ply')


@_deprecate_positional_args
def download_pine_roots(load=True):  # noqa: FBT002
    """Download pine roots dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_pine_roots()
    >>> dataset.plot()

    .. seealso::

        :ref:`Pine Roots Dataset <pine_roots_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`connectivity_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_pine_roots, load=load)


_dataset_pine_roots = _SingleFileDownloadableDatasetLoader('pine_root.tri')


@_deprecate_positional_args
def download_crater_topo(load=True):  # noqa: FBT002
    """Download crater dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ImageData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_crater_topo()
    >>> dataset.plot(cmap='gist_earth', cpos='xy')

    .. seealso::

        :ref:`Crater Topo Dataset <crater_topo_dataset>`
            See this dataset in the Dataset Gallery for more info.

        This dataset is used in the following examples:

        * :ref:`terrain_following_mesh_example`
        * :ref:`topo_map_example`

    """
    return _download_dataset(_dataset_crater_topo, load=load)


_dataset_crater_topo = _SingleFileDownloadableDatasetLoader('Ruapehu_mag_dem_15m_NZTM.vtk')


@_deprecate_positional_args
def download_crater_imagery(load=True):  # noqa: FBT002
    """Download crater texture.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.Texture | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> cpos = [
    ...     [66.0, 73.0, -382.6],
    ...     [66.0, 73.0, 0.0],
    ...     [-0.0, -1.0, 0.0],
    ... ]
    >>> texture = examples.download_crater_imagery()
    >>> texture.plot(cpos=cpos)

    .. seealso::

        :ref:`Crater Imagery Dataset <crater_imagery_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`topo_map_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_crater_imagery, load=load)


_dataset_crater_imagery = _SingleFileDownloadableDatasetLoader(
    'BJ34_GeoTifv1-04_crater_clip.tif',
    read_func=read_texture,  # type: ignore[arg-type]
)


@_deprecate_positional_args
def download_dolfin(load=True):  # noqa: FBT002
    """Download dolfin mesh.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_dolfin()
    >>> dataset.plot(cpos='xy', show_edges=True)

    .. seealso::

        :ref:`Dolfin Dataset <dolfin_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`read_dolfin_example`

    """
    return _download_dataset(_dataset_dolfin, load=load)


_dataset_dolfin = _SingleFileDownloadableDatasetLoader(
    'dolfin_fine.xml',
    read_func=functools.partial(read, file_format='dolfin-xml'),  # type: ignore[arg-type]
)


@_deprecate_positional_args
def download_damavand_volcano(load=True):  # noqa: FBT002
    """Download damavand volcano model.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ImageData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Load the dataset.

    >>> from pyvista import examples
    >>> dataset = examples.download_damavand_volcano()

    Use :meth:`~pyvista.ImageDataFilters.resample` to downsample it before plotting.

    >>> dataset = dataset.resample(0.5)
    >>> dataset.dimensions
    (140, 116, 85)

    Plot it.

    >>> cpos = [
    ...     [4.66316700e04, 4.32796241e06, -3.82467050e05],
    ...     [5.52532740e05, 3.98017300e06, -2.47450000e04],
    ...     [4.10000000e-01, -2.90000000e-01, -8.60000000e-01],
    ... ]
    >>> dataset.plot(cpos=cpos, cmap='reds', show_scalar_bar=False, volume=True)

    .. seealso::

        :ref:`Damavand Volcano Dataset <damavand_volcano_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`volume_rendering_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_damavand_volcano, load=load)


def _damavand_volcano_load_func(volume):
    volume.rename_array('None', 'data')
    return volume


_dataset_damavand_volcano = _SingleFileDownloadableDatasetLoader(
    'damavand-volcano.vtk',
    load_func=_damavand_volcano_load_func,
)


@_deprecate_positional_args
def download_delaunay_example(load=True):  # noqa: FBT002
    """Download a pointset for the Delaunay example.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_delaunay_example()
    >>> dataset.plot(show_edges=True)

    .. seealso::

        :ref:`Delaunay Example Dataset <delaunay_example_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_delaunay_example, load=load)


_dataset_delaunay_example = _SingleFileDownloadableDatasetLoader('250.vtk')


@_deprecate_positional_args
def download_embryo(load=True):  # noqa: FBT002
    """Download a volume of an embryo.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ImageData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_embryo()
    >>> dataset.plot(volume=True)

    .. seealso::

        :ref:`Embryo Dataset <embryo_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`medical_dataset_gallery`
            Browse other medical datasets.

        This dataset is used in the following examples:

        * :ref:`contouring_example`
        * :ref:`resampling_example`
        * :ref:`slice_orthogonal_example`

    """
    return _download_dataset(_dataset_embryo, load=load)


def _embryo_load_func(dataset):
    # cleanup artifact
    mask = dataset['SLCImage'] == 255
    dataset['SLCImage'][mask] = 0
    return dataset


_dataset_embryo = _SingleFileDownloadableDatasetLoader('embryo.slc', load_func=_embryo_load_func)


@_deprecate_positional_args
def download_antarctica_velocity(load=True):  # noqa: FBT002
    """Download the antarctica velocity simulation results.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_antarctica_velocity()
    >>> dataset.plot(cpos='xy', clim=[1e-3, 1e4], cmap='Blues', log_scale=True)

    .. seealso::

        :ref:`Antarctica Velocity Dataset <antarctica_velocity_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`antarctica_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_antarctica_velocity, load=load)


_dataset_antarctica_velocity = _SingleFileDownloadableDatasetLoader('antarctica_velocity.vtp')


@_deprecate_positional_args
def download_room_surface_mesh(load=True):  # noqa: FBT002
    """Download the room surface mesh.

    This mesh is for demonstrating the difference that depth peeling can
    provide when rendering translucent geometries.

    This mesh is courtesy of `Sam Potter <https://github.com/sampotter>`_.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_room_surface_mesh()
    >>> dataset.plot()

    .. seealso::

        :ref:`Room Surface Mesh Dataset <room_surface_mesh_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`depth_peeling_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_room_surface_mesh, load=load)


_dataset_room_surface_mesh = _SingleFileDownloadableDatasetLoader('room_surface_mesh.obj')


@_deprecate_positional_args
def download_beach(load=True):  # noqa: FBT002
    """Download the beach NRRD image.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ImageData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_beach()
    >>> dataset.plot(rgba=True, cpos='xy')

    .. seealso::

        :ref:`Beach Dataset <beach_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_beach, load=load)


_dataset_beach = _SingleFileDownloadableDatasetLoader('beach.nrrd')


@_deprecate_positional_args
def download_rgba_texture(load=True):  # noqa: FBT002
    """Download a texture with an alpha channel.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.Texture | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_rgba_texture()
    >>> dataset.plot(cpos='xy')

    .. seealso::

        :ref:`Rgba Texture Dataset <rgba_texture_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`texture_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_rgba_texture, load=load)


_dataset_rgba_texture = _SingleFileDownloadableDatasetLoader(
    'alphachannel.png',
    read_func=read_texture,  # type: ignore[arg-type]
)


@_deprecate_positional_args
def download_vtk_logo(load=True):  # noqa: FBT002
    """Download a texture of the VTK logo.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.Texture | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_vtk_logo()
    >>> dataset.plot(cpos='xy')

    .. seealso::

        :ref:`Vtk Logo Dataset <vtk_logo_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Vtk Dataset <vtk_dataset>`

    """
    return _download_dataset(_dataset_vtk_logo, load=load)


_dataset_vtk_logo = _SingleFileDownloadableDatasetLoader('vtk.png', read_func=read_texture)  # type: ignore[arg-type]


@_deprecate_positional_args
def download_sky_box_cube_map(load=True):  # noqa: FBT002
    """Download a skybox cube map texture.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.Texture
        Texture containing a skybox.

    Examples
    --------
    >>> from pyvista import examples
    >>> import pyvista as pv
    >>> pl = pv.Plotter()
    >>> dataset = examples.download_sky_box_cube_map()
    >>> _ = pl.add_actor(dataset.to_skybox())
    >>> pl.set_environment_texture(dataset)
    >>> pl.show()

    .. seealso::

        :ref:`Sky Box Cube Map Dataset <sky_box_cube_map_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Cubemap Space 4k Dataset <cubemap_space_4k_dataset>`

        :ref:`Cubemap Space 16k Dataset <cubemap_space_16k_dataset>`

        :ref:`Cubemap Park Dataset <cubemap_park_dataset>`

        :ref:`pbr_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_sky_box_cube_map, load=load)


def _sky_box_cube_map_files_func():
    posx = _DownloadableFile(
        'skybox2-posx.jpg',
    )
    negx = _DownloadableFile('skybox2-negx.jpg')
    posy = _DownloadableFile('skybox2-posy.jpg')
    negy = _DownloadableFile('skybox2-negy.jpg')
    posz = _DownloadableFile('skybox2-posz.jpg')
    negz = _DownloadableFile('skybox2-negz.jpg')
    return posx, negx, posy, negy, posz, negz


_dataset_sky_box_cube_map = _MultiFileDownloadableDatasetLoader(
    files_func=_sky_box_cube_map_files_func,
    load_func=_load_as_cubemap,
)


@_deprecate_positional_args
def download_cubemap_park(load=True):  # noqa: FBT002
    """Download a cubemap of a park.

    Downloaded from http://www.humus.name/index.php?page=Textures
    by David Eck, and converted to a smaller 512x512 size for use
    with WebGL in his free, on-line textbook at
    http://math.hws.edu/graphicsbook

    This work is licensed under a Creative Commons Attribution 3.0 Unported
    License.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.Texture
        Texture containing a skybox.

    Examples
    --------
    >>> from pyvista import examples
    >>> import pyvista as pv
    >>> pl = pv.Plotter(lighting=None)
    >>> dataset = examples.download_cubemap_park()
    >>> _ = pl.add_actor(dataset.to_skybox())
    >>> pl.set_environment_texture(dataset, is_srgb=True)
    >>> pl.camera_position = 'xy'
    >>> pl.camera.zoom(0.4)
    >>> _ = pl.add_mesh(pv.Sphere(), pbr=True, roughness=0.1, metallic=0.5)
    >>> pl.show()

    .. seealso::

        :ref:`Cubemap Park Dataset <cubemap_park_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Cubemap Space 4k Dataset <cubemap_space_4k_dataset>`

        :ref:`Cubemap Space 16k Dataset <cubemap_space_16k_dataset>`

        :ref:`Sky Box Cube Map Dataset <sky_box_cube_map_dataset>`

    """
    return _download_dataset(_dataset_cubemap_park, load=load)


_dataset_cubemap_park = _SingleFileDownloadableDatasetLoader(
    'cubemap_park/cubemap_park.zip',
    target_file='',
    read_func=_load_as_cubemap,  # type: ignore[arg-type]
)


@_deprecate_positional_args
def download_cubemap_space_4k(load=True):  # noqa: FBT002
    """Download the 4k space cubemap.

    This cubemap was generated by downloading the 4k image from: `Deep Star
    Maps 2020 <https://svs.gsfc.nasa.gov/4851>`_ and converting it using
    https://jaxry.github.io/panorama-to-cubemap/

    See `vtk-data/cubemap_space
    <https://github.com/pyvista/vtk-data/tree/master/Data/cubemap_space#readme>`_
    for more details.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.Texture
        Texture containing a skybox.

    Examples
    --------
    Display the cubemap as both an environment texture and an actor.

    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> cubemap = examples.download_cubemap_space_4k()
    >>> pl = pv.Plotter(lighting=None)
    >>> _ = pl.add_actor(cubemap.to_skybox())
    >>> pl.set_environment_texture(cubemap, is_srgb=True)
    >>> pl.camera.zoom(0.4)
    >>> _ = pl.add_mesh(pv.Sphere(), pbr=True, roughness=0.24, metallic=1.0)
    >>> pl.show()

    .. seealso::

        :ref:`Cubemap Space 4k Dataset <cubemap_space_4k_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Cubemap Space 16k Dataset <cubemap_space_16k_dataset>`

        :ref:`Cubemap Park Dataset <cubemap_park_dataset>`

        :ref:`Sky Box Cube Map Dataset <sky_box_cube_map_dataset>`

    """
    return _download_dataset(_dataset_cubemap_space_4k, load=load)


_dataset_cubemap_space_4k = _SingleFileDownloadableDatasetLoader(
    'cubemap_space/4k.zip',
    target_file='',
    read_func=_load_as_cubemap,  # type: ignore[arg-type]
)


@_deprecate_positional_args
def download_cubemap_space_16k(load=True):  # noqa: FBT002
    """Download the 16k space cubemap.

    This cubemap was generated by downloading the 16k image from: `Deep Star
    Maps 2020 <https://svs.gsfc.nasa.gov/4851>`_ and converting it using
    https://jaxry.github.io/panorama-to-cubemap/

    See `vtk-data/cubemap_space
    <https://github.com/pyvista/vtk-data/tree/master/Data/cubemap_space#readme>`_ for
    more details.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.Texture
        Texture containing a skybox.

    Notes
    -----
    This is a 38MB file and may take a while to download.

    Examples
    --------
    Display the cubemap as both an environment texture and an actor. Note that
    here we're displaying the 4k as the 16k is a bit too expensive to display
    in the documentation.

    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> cubemap = examples.download_cubemap_space_4k()
    >>> pl = pv.Plotter(lighting=None)
    >>> _ = pl.add_actor(cubemap.to_skybox())
    >>> pl.set_environment_texture(cubemap, is_srgb=True)
    >>> pl.camera.zoom(0.4)
    >>> _ = pl.add_mesh(pv.Sphere(), pbr=True, roughness=0.24, metallic=1.0)
    >>> pl.show()

    .. seealso::

        :ref:`Cubemap Space 16k Dataset <cubemap_space_16k_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Cubemap Space 4k Dataset <cubemap_space_4k_dataset>`

        :ref:`Cubemap Park Dataset <cubemap_park_dataset>`

        :ref:`Sky Box Cube Map Dataset <sky_box_cube_map_dataset>`

    """
    return _download_dataset(_dataset_cubemap_space_16k, load=load)


_dataset_cubemap_space_16k = _SingleFileDownloadableDatasetLoader(
    'cubemap_space/16k.zip',
    target_file='',
    read_func=_load_as_cubemap,  # type: ignore[arg-type]
)


@_deprecate_positional_args
def download_backward_facing_step(load=True):  # noqa: FBT002
    """Download an ensight gold case of a fluid simulation.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.MultiBlock | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_backward_facing_step()
    >>> dataset.plot()

    .. seealso::

        :ref:`Backward Facing Step Dataset <backward_facing_step_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_backward_facing_step, load=load)


_dataset_backward_facing_step = _SingleFileDownloadableDatasetLoader(
    'EnSight.zip',
    target_file='foam_case_0_0_0_0.case',
)


@_deprecate_positional_args
def download_gpr_data_array(load=True):  # noqa: FBT002
    """Download GPR example data array.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    numpy.ndarray | str
        Array or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_gpr_data_array()  # doctest:+SKIP
    >>> dataset  # doctest:+SKIP
    array([[nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan],
           [nan, nan, nan, ..., nan, nan, nan],
           ...,
           [ 0.,  0.,  0., ...,  0.,  0.,  0.],
           [ 0.,  0.,  0., ...,  0.,  0.,  0.],
           [ 0.,  0.,  0., ...,  0.,  0.,  0.]])

    .. seealso::

        :ref:`Gpr Data Array Dataset <gpr_data_array_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Gpr Path Dataset <gpr_path_dataset>`

        :ref:`create_draped_surface_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_gpr_data_array, load=load)


_dataset_gpr_data_array = _SingleFileDownloadableDatasetLoader(
    'gpr-example/data.npy',
    read_func=np.load,
)


@_deprecate_positional_args
def download_gpr_path(load=True):  # noqa: FBT002
    """Download GPR example path.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_gpr_path()
    >>> dataset.plot()

    .. seealso::

        :ref:`Gpr Path Dataset <gpr_path_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Gpr Data Array Dataset <gpr_data_array_dataset>`

        :ref:`create_draped_surface_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_gpr_path, load=load)


_dataset_gpr_path = _SingleFileDownloadableDatasetLoader(
    'gpr-example/path.txt',
    read_func=functools.partial(np.loadtxt, skiprows=1),  # type: ignore[arg-type]
    load_func=pyvista.PolyData,  # type: ignore[arg-type]
)


@_deprecate_positional_args
def download_woman(load=True):  # noqa: FBT002
    """Download scan of a woman.

    Originally obtained from Laser Design.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_woman()
    >>> cpos = [
    ...     (-2600.0, 1970.6, 1836.9),
    ...     (48.5, -20.3, 843.9),
    ...     (0.23, -0.168, 0.958),
    ... ]
    >>> dataset.plot(cpos=cpos)

    .. seealso::

        :ref:`Woman Dataset <woman_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_woman, load=load)


_dataset_woman = _SingleFileDownloadableDatasetLoader('woman.stl')


@_deprecate_positional_args
def download_lobster(load=True):  # noqa: FBT002
    """Download scan of a lobster.

    Originally obtained from Laser Design.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_lobster()
    >>> dataset.plot()

    .. seealso::

        :ref:`Lobster Dataset <lobster_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_lobster, load=load)


_dataset_lobster = _SingleFileDownloadableDatasetLoader('lobster.ply')


@_deprecate_positional_args
def download_face2(load=True):  # noqa: FBT002
    """Download scan of a man's face.

    Originally obtained from Laser Design.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_face2()
    >>> dataset.plot()

    .. seealso::

        :ref:`Face2 Dataset <face2_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Face Dataset <face_dataset>`

    """
    return _download_dataset(_dataset_face2, load=load)


_dataset_face2 = _SingleFileDownloadableDatasetLoader('man_face.stl')


@_deprecate_positional_args
def download_urn(load=True):  # noqa: FBT002
    """Download scan of a burial urn.

    Originally obtained from Laser Design.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> cpos = [
    ...     [-7.123e02, 5.715e02, 8.601e02],
    ...     [4.700e00, 2.705e02, -1.010e01],
    ...     [2.000e-01, 1.000e00, -2.000e-01],
    ... ]
    >>> dataset = examples.download_urn()
    >>> dataset.plot(cpos=cpos)

    .. seealso::

        :ref:`Urn Dataset <urn_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_urn, load=load)


_dataset_urn = _SingleFileDownloadableDatasetLoader('urn.stl')


@_deprecate_positional_args
def download_pepper(load=True):  # noqa: FBT002
    """Download scan of a pepper (capsicum).

    Originally obtained from Laser Design.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_pepper()
    >>> dataset.plot()

    .. seealso::

        :ref:`Pepper Dataset <pepper_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_pepper, load=load)


_dataset_pepper = _SingleFileDownloadableDatasetLoader('pepper.ply')


@_deprecate_positional_args
def download_drill(load=True):  # noqa: FBT002
    """Download scan of a power drill.

    Originally obtained from Laser Design.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_drill()
    >>> dataset.plot()

    .. seealso::

        :ref:`Drill Dataset <drill_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    # Silence warning: unexpected data at end of line in OBJ file
    with pyvista.vtk_verbosity('off'):
        return _download_dataset(_dataset_drill, load=load)


_dataset_drill = _SingleFileDownloadableDatasetLoader('drill.obj')


@_deprecate_positional_args
def download_action_figure(load=True, *, high_resolution=False):  # noqa: FBT002
    """Download scan of an action figure.

    Originally obtained from Laser Design.

    .. versionchanged:: 0.45

        A decimated version of this dataset with 31 thousand cells is now returned.
        Previously, a high-resolution version with 630 thousand cells was returned.
        Use ``high_resolution=True`` for the high-resolution version.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    high_resolution : bool, default: False
        Set this to ``True`` to return a high-resolution version of this dataset.
        By default, a :meth:`decimated <pyvista.PolyDataFilters.decimate>` version
        is returned with 95% reduction.

        .. versionadded:: 0.45

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Show the action figure example. This also demonstrates how to use
    physically based rendering and lighting to make a good looking
    plot.

    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> dataset = examples.download_action_figure()
    >>> _ = dataset.clean(inplace=True)
    >>> pl = pv.Plotter(lighting=None)
    >>> pl.add_light(pv.Light(position=(30, 10, 10)))
    >>> _ = pl.add_mesh(
    ...     dataset,
    ...     color='w',
    ...     smooth_shading=True,
    ...     pbr=True,
    ...     metallic=0.3,
    ...     roughness=0.5,
    ... )
    >>> pl.camera_position = [
    ...     (32.3, 116.3, 220.6),
    ...     (-0.05, 3.8, 33.8),
    ...     (-0.017, 0.86, -0.51),
    ... ]
    >>> pl.show()

    .. seealso::

        :ref:`Action Figure Dataset <action_figure_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    if high_resolution:
        return _download_dataset(__dataset_action_figure_high_res, load=load)
    return _download_dataset(_dataset_action_figure, load=load)


_dataset_action_figure = _SingleFileDownloadableDatasetLoader('tigerfighter_decimated.obj')
__dataset_action_figure_high_res = _SingleFileDownloadableDatasetLoader('tigerfighter.obj')


@_deprecate_positional_args
def download_notch_stress(load=True):  # noqa: FBT002
    """Download the FEA stress result from a notched beam.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_notch_stress()
    >>> dataset.plot(cmap='bwr')

    .. seealso::

        :ref:`Notch Stress Dataset <notch_stress_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Notch Displacement Dataset <notch_displacement_dataset>`

        :ref:`Aero Bracket Dataset <aero_bracket_dataset>`

        :ref:`Fea Bracket Dataset <fea_bracket_dataset>`

        :ref:`Fea Hertzian Contact Cylinder Dataset <fea_hertzian_contact_cylinder_dataset>`

    """
    return _download_dataset(_dataset_notch_stress, load=load)


_dataset_notch_stress = _SingleFileDownloadableDatasetLoader('notch_stress.vtk')


@_deprecate_positional_args
def download_notch_displacement(load=True):  # noqa: FBT002
    """Download the FEA displacement result from a notched beam.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_notch_displacement()
    >>> dataset.plot(cmap='bwr')

    .. seealso::

        :ref:`Notch Displacement Dataset <notch_displacement_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Notch Stress Dataset <notch_stress_dataset>`

        :ref:`Aero Bracket Dataset <aero_bracket_dataset>`

        :ref:`Fea Bracket Dataset <fea_bracket_dataset>`

        :ref:`Fea Hertzian Contact Cylinder Dataset <fea_hertzian_contact_cylinder_dataset>`

    """
    return _download_dataset(_dataset_notch_displacement, load=load)


_dataset_notch_displacement = _SingleFileDownloadableDatasetLoader('notch_disp.vtu')


@_deprecate_positional_args
def download_louis_louvre(load=True):  # noqa: FBT002
    """Download the Louis XIV de France statue at the Louvre, Paris.

    Statue found in the Napoléon Courtyard of Louvre Palace. It is a
    copy in plomb of the original statue in Versailles, made by
    Bernini and Girardon.

    Originally downloaded from `sketchfab <https://sketchfab.com/3d-models/louis-xiv-de-france-louvre-paris-a0cc0e7eee384c99838dff2857b8158c>`_

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Plot the Louis XIV statue with custom lighting and camera angle.

    >>> from pyvista import examples
    >>> import pyvista as pv
    >>> dataset = examples.download_louis_louvre()
    >>> pl = pv.Plotter(lighting=None)
    >>> _ = pl.add_mesh(dataset, smooth_shading=True)
    >>> pl.add_light(pv.Light(position=(10, -10, 10)))
    >>> pl.camera_position = [
    ...     [-6.71, -14.55, 15.17],
    ...     [1.44, 2.54, 9.84],
    ...     [0.16, 0.22, 0.96],
    ... ]
    >>> pl.show()

    .. seealso::

        :ref:`Louis Louvre Dataset <louis_louvre_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`pbr_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_louis_louvre, load=load)


_dataset_louis_louvre = _SingleFileDownloadableDatasetLoader('louis.ply')


@_deprecate_positional_args
def download_cylinder_crossflow(load=True):  # noqa: FBT002
    """Download CFD result for cylinder in cross flow at Re=35.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.MultiBlock | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_cylinder_crossflow()
    >>> dataset.plot(cpos='xy', cmap='blues', rng=[-200, 500])

    .. seealso::

        :ref:`Cylinder Crossflow Dataset <cylinder_crossflow_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`streamlines_2D_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_cylinder_crossflow, load=load)


def _cylinder_crossflow_files_func():
    case = _SingleFileDownloadableDatasetLoader('EnSight/CylinderCrossflow/cylinder_Re35.case')
    geo = _DownloadableFile('EnSight/CylinderCrossflow/cylinder_Re35.geo')
    scl1 = _DownloadableFile('EnSight/CylinderCrossflow/cylinder_Re35.scl1')
    scl2 = _DownloadableFile('EnSight/CylinderCrossflow/cylinder_Re35.scl2')
    vel = _DownloadableFile('EnSight/CylinderCrossflow/cylinder_Re35.vel')
    return case, geo, scl1, scl2, vel


_dataset_cylinder_crossflow = _MultiFileDownloadableDatasetLoader(
    files_func=_cylinder_crossflow_files_func,
)


@_deprecate_positional_args
def download_naca(load=True):  # noqa: FBT002
    """Download NACA airfoil dataset in EnSight format.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.MultiBlock | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Plot the density of the air surrounding the NACA airfoil using the
    ``"jet"`` color map.

    >>> from pyvista import examples
    >>> cpos = [[-0.22, 0.0, 2.52], [0.43, 0.0, 0.0], [0.0, 1.0, 0.0]]
    >>> dataset = examples.download_naca()
    >>> dataset.plot(cpos=cpos, cmap='jet')

    .. seealso::

        :ref:`Naca Dataset <naca_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`reader_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_naca, load=load)


def _naca_files_func():
    case = _SingleFileDownloadableDatasetLoader('EnSight/naca.bin.case')
    dens1 = _DownloadableFile('EnSight/naca.gold.bin.DENS_1')
    dens3 = _DownloadableFile('EnSight/naca.gold.bin.DENS_3')
    geo = _DownloadableFile('EnSight/naca.gold.bin.geo')
    return case, dens1, dens3, geo


_dataset_naca = _MultiFileDownloadableDatasetLoader(files_func=_naca_files_func)


@_deprecate_positional_args
def download_lshape(load=True):  # noqa: FBT002
    """Download LShape dataset in EnSight format.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.MultiBlock | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Load and plot the dataset.

    >>> from pyvista import examples
    >>> mesh = examples.download_lshape()['all']
    >>> warped = mesh.warp_by_vector(factor=30)
    >>> warped.plot(scalars='displacement')

    .. seealso::

        :ref:`Lshape Dataset <lshape_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_lshape, load=load)


def _lshape_files_func():
    def read_func(filename):
        reader = pyvista.get_reader(filename)
        reader.set_active_time_set(1)
        reader.set_active_time_value(1.0)
        return reader.read()

    case = _SingleFileDownloadableDatasetLoader('EnSight/LShape.case', read_func=read_func)
    geo = _DownloadableFile('EnSight/LShape_geometry.geo')
    var = _DownloadableFile('EnSight/LShape_displacement.var')
    return case, geo, var


_dataset_lshape = _MultiFileDownloadableDatasetLoader(files_func=_lshape_files_func)


@_deprecate_positional_args
def download_wavy(load=True):  # noqa: FBT002
    """Download PVD file of a 2D wave.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.MultiBlock | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_wavy()
    >>> dataset.plot()

    .. seealso::

        :ref:`Wavy Dataset <wavy_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`reader_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_wavy, load=load)


_dataset_wavy = _SingleFileDownloadableDatasetLoader('PVD/wavy.zip', target_file='unzip/wavy.pvd')


@_deprecate_positional_args
def download_single_sphere_animation(load=True):  # noqa: FBT002
    """Download PVD file for single sphere.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.MultiBlock | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> import os
    >>> from tempfile import mkdtemp
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> filename = examples.download_single_sphere_animation(load=False)
    >>> reader = pv.PVDReader(filename)

    Write the gif to a temporary directory. Normally you would write to a local
    path.

    >>> gif_filename = os.path.join(mkdtemp(), 'single_sphere.gif')

    Generate the animation.

    >>> plotter = pv.Plotter()
    >>> plotter.open_gif(gif_filename)
    >>> for time_value in reader.time_values:
    ...     reader.set_active_time_value(time_value)
    ...     mesh = reader.read()
    ...     _ = plotter.add_mesh(mesh, smooth_shading=True)
    ...     _ = plotter.add_text(f'Time: {time_value:.0f}', color='black')
    ...     plotter.write_frame()
    ...     plotter.clear()
    ...     plotter.enable_lightkit()
    >>> plotter.close()

    .. seealso::

        :ref:`Single Sphere Animation Dataset <single_sphere_animation_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Dual Sphere Animation Dataset <dual_sphere_animation_dataset>`

    """
    return _download_dataset(_dataset_single_sphere_animation, load=load)


_dataset_single_sphere_animation = _SingleFileDownloadableDatasetLoader(
    'PVD/paraview/singleSphereAnimation.zip',
    target_file='singleSphereAnimation.pvd',
)


@_deprecate_positional_args
def download_dual_sphere_animation(load=True):  # noqa: FBT002
    """Download PVD file for double sphere.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.MultiBlock | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> import os
    >>> from tempfile import mkdtemp
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> filename = examples.download_dual_sphere_animation(load=False)
    >>> reader = pv.PVDReader(filename)

    Write the gif to a temporary directory. Normally you would write to a local
    path.

    >>> gif_filename = os.path.join(mkdtemp(), 'dual_sphere.gif')

    Generate the animation.

    >>> plotter = pv.Plotter()
    >>> plotter.open_gif(gif_filename)
    >>> for time_value in reader.time_values:
    ...     reader.set_active_time_value(time_value)
    ...     mesh = reader.read()
    ...     _ = plotter.add_mesh(mesh, smooth_shading=True)
    ...     _ = plotter.add_text(f'Time: {time_value:.0f}', color='black')
    ...     plotter.write_frame()
    ...     plotter.clear()
    ...     plotter.enable_lightkit()
    >>> plotter.close()

    .. seealso::

        :ref:`Dual Sphere Animation Dataset <dual_sphere_animation_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Single Sphere Animation Dataset <single_sphere_animation_dataset>`

    """
    return _download_dataset(_dataset_dual_sphere_animation, load=load)


_dataset_dual_sphere_animation = _SingleFileDownloadableDatasetLoader(
    'PVD/paraview/dualSphereAnimation.zip',
    target_file='dualSphereAnimation.pvd',
)


@_deprecate_positional_args
def download_cavity(load=True):  # noqa: FBT002
    """Download cavity OpenFOAM example.

    Retrieved from
    `Kitware VTK Data <https://data.kitware.com/#collection/55f17f758d777f6ddc7895b7/folder/5afd932e8d777f15ebe1b183>`_.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.MultiBlock | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_cavity()  # doctest:+SKIP

    .. seealso::

        :ref:`Cavity Dataset <cavity_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`openfoam_example`
            Full example using this dataset.

    """
    return _download_dataset(_dataset_cavity, load=load)


_dataset_cavity = _SingleFileDownloadableDatasetLoader(
    'OpenFOAM.zip',
    target_file='cavity/case.foam',
)


@_deprecate_positional_args
def download_openfoam_tubes(load=True):  # noqa: FBT002
    """Download tubes OpenFOAM example.

    Data generated from public SimScale examples at `SimScale Project Library -
    Turbo <https://www.simscale.com/projects/ayarnoz/turbo/>`_.

    Licensing for this dataset is granted to freely and without restriction
    reproduce, distribute, publish according to the `SimScale Terms and
    Conditions <https://www.simscale.com/terms-and-conditions/>`_.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.MultiBlock | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Plot the outline of the dataset along with a cross section of the flow velocity.

    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> dataset = examples.download_openfoam_tubes()
    >>> air = dataset[0]
    >>> y_slice = air.slice('y')
    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(
    ...     y_slice,
    ...     scalars='U',
    ...     lighting=False,
    ...     scalar_bar_args={'title': 'Flow Velocity'},
    ... )
    >>> _ = pl.add_mesh(air, color='w', opacity=0.25)
    >>> pl.enable_anti_aliasing()
    >>> pl.show()

    .. seealso::

        :ref:`Openfoam Tubes Dataset <openfoam_tubes_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`openfoam_tubes_example`
            Full example using this dataset.

    """
    return _download_dataset(_dataset_openfoam_tubes, load=load)


def _openfoam_tubes_read_func(filename):
    reader = pyvista.OpenFOAMReader(filename)
    reader.set_active_time_value(1000)
    return reader.read()


_dataset_openfoam_tubes = _SingleFileDownloadableDatasetLoader(
    'fvm/turbo_incompressible/Turbo-Incompressible_3-Run_1-SOLUTION_FIELDS.zip',
    target_file='case.foam',
    read_func=_openfoam_tubes_read_func,
)


@_deprecate_positional_args
def download_lucy(load=True):  # noqa: FBT002
    """Download the lucy angel mesh.

    Original downloaded from the `The Stanford 3D Scanning Repository
    <http://graphics.stanford.edu/data/3Dscanrep/>`_ and decimated to
    approximately 100k triangle.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Plot the Lucy Angel dataset with custom lighting.

    >>> from pyvista import examples
    >>> import pyvista as pv
    >>> dataset = examples.download_lucy()

    Create a light at the "flame".

    >>> flame_light = pv.Light(
    ...     color=[0.886, 0.345, 0.133],
    ...     position=[550, 140, 950],
    ...     intensity=1.5,
    ...     positional=True,
    ...     cone_angle=90,
    ...     attenuation_values=(0.001, 0.005, 0),
    ... )

    Create a scene light.

    >>> scene_light = pv.Light(intensity=0.2)

    >>> pl = pv.Plotter(lighting=None)
    >>> _ = pl.add_mesh(dataset, smooth_shading=True)
    >>> pl.add_light(flame_light)
    >>> pl.add_light(scene_light)
    >>> pl.background_color = 'k'
    >>> pl.show()

    .. seealso::

        :ref:`Lucy Dataset <lucy_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`jupyter_plotting`
            Example using this dataset.

    """
    return _download_dataset(_dataset_lucy, load=load)


_dataset_lucy = _SingleFileDownloadableDatasetLoader('lucy.ply')


@_deprecate_positional_args
def download_pump_bracket(load=True):  # noqa: FBT002
    """Download the pump bracket example dataset.

    Data generated from public SimScale examples at `SimScale Project Library -
    Turbo <https://www.simscale.com/projects/STR/bracket/>`_.

    Licensing for this dataset is granted freely and without restriction to
    reproduce, distribute, and publish according to the `SimScale Terms and
    Conditions <https://www.simscale.com/terms-and-conditions/>`_.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    UnstructuredGrid | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Load the dataset.

    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> dataset = examples.download_pump_bracket()
    >>> dataset
    UnstructuredGrid (...)
      N Cells:    124806
      N Points:   250487
      X Bounds:   -5.000e-01, 5.000e-01
      Y Bounds:   -4.000e-01, 0.000e+00
      Z Bounds:   -2.500e-02, 2.500e-02
      N Arrays:   10

    Plot the displacement of the 4th mode shape as scalars.

    >>> cpos = [
    ...     (0.744, -0.502, -0.830),
    ...     (0.0520, -0.160, 0.0743),
    ...     (-0.180, -0.958, 0.224),
    ... ]
    >>> dataset.plot(
    ...     scalars='disp_3',
    ...     cpos=cpos,
    ...     show_scalar_bar=False,
    ...     ambient=0.2,
    ...     anti_aliasing='fxaa',
    ... )

    .. seealso::

        :ref:`Pump Bracket Dataset <pump_bracket_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`pump_bracket_example`
            Full example using this dataset.

    """
    return _download_dataset(_dataset_pump_bracket, load=load)


_dataset_pump_bracket = _SingleFileDownloadableDatasetLoader(
    'fea/pump_bracket/pump_bracket.zip',
    target_file='pump_bracket.vtk',
)


@_deprecate_positional_args
def download_electronics_cooling(load=True):  # noqa: FBT002
    """Download the electronics cooling example datasets.

    Data generated from public SimScale examples at `SimScale Project Library -
    Turbo <https://www.simscale.com/projects/ayarnoz/turbo/>`_.

    Licensing for this dataset is granted to freely and without restriction
    reproduce, distribute, publish according to the `SimScale Terms and
    Conditions <https://www.simscale.com/terms-and-conditions/>`_.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    tuple[PolyData, UnstructuredGrid] | list[str]
        DataSets or filenames depending on ``load``.

    Examples
    --------
    Load the datasets and plot the air velocity through the electronics.

    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> structure, air = examples.download_electronics_cooling()

    Show the type and bounds of the datasets.

    >>> structure, air
    (PolyData (...)
      N Cells:    344270
      N Points:   187992
      N Strips:   0
      X Bounds:   -3.000e-03, 1.530e-01
      Y Bounds:   -3.000e-03, 2.030e-01
      Z Bounds:   -9.000e-03, 4.200e-02
      N Arrays:   4, UnstructuredGrid (...)
      N Cells:    1749992
      N Points:   610176
      X Bounds:   -1.388e-18, 1.500e-01
      Y Bounds:   -3.000e-03, 2.030e-01
      Z Bounds:   -6.000e-03, 4.400e-02
      N Arrays:   10)

    >>> z_slice = air.clip('z', value=-0.005)
    >>> pl = pv.Plotter()
    >>> pl.enable_ssao(radius=0.01)
    >>> _ = pl.add_mesh(
    ...     z_slice,
    ...     scalars='U',
    ...     lighting=False,
    ...     scalar_bar_args={'title': 'Velocity'},
    ... )
    >>> _ = pl.add_mesh(
    ...     structure,
    ...     color='w',
    ...     smooth_shading=True,
    ...     split_sharp_edges=True,
    ... )
    >>> pl.camera_position = 'xy'
    >>> pl.camera.roll = 90
    >>> pl.enable_anti_aliasing('fxaa')
    >>> pl.show()

    .. seealso::

        :ref:`Electronics Cooling Dataset <electronics_cooling_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`openfoam_cooling_example`
            Full example using this dataset.

    """
    return _download_dataset(_dataset_electronics_cooling, load=load)


def _electronics_cooling_files_func():
    _structure = _SingleFileDownloadableDatasetLoader(
        'fvm/cooling_electronics/datasets.zip',
        target_file='structure.vtp',
    )
    _air = _SingleFileDownloadableDatasetLoader(
        'fvm/cooling_electronics/datasets.zip',
        target_file='air.vtu',
    )
    return _structure, _air


_dataset_electronics_cooling = _MultiFileDownloadableDatasetLoader(
    _electronics_cooling_files_func,
    load_func=_load_as_multiblock,
)


@_deprecate_positional_args
def download_can_crushed_hdf(load=True):  # noqa: FBT002
    """Download the crushed can dataset.

    File obtained from `Kitware <https://www.kitware.com/>`_. Used
    for testing hdf files.

    Originally built using VTK v9.2.0rc from:

    ``VTK/build/ExternalTesting/can-vtu.hdf``

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid | str
        Crushed can dataset or path depending on the value of ``load``.

    Examples
    --------
    Plot the crushed can dataset.

    >>> from pyvista import examples
    >>> import pyvista as pv
    >>> dataset = examples.download_can_crushed_hdf()
    >>> dataset.plot(smooth_shading=True)

    .. seealso::

        :ref:`Can Crushed Hdf Dataset <can_crushed_hdf_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Can Crushed Vtu Dataset <can_crushed_vtu_dataset>`

    """
    return _download_dataset(_dataset_can_crushed_hdf, load=load)


_dataset_can_crushed_hdf = _SingleFileDownloadableDatasetLoader('hdf/can-vtu.hdf')


@_deprecate_positional_args
def download_can_crushed_vtu(load=True):  # noqa: FBT002
    """Download the crushed can dataset.

    File obtained from `Kitware <https://www.kitware.com/>`_. Used
    for testing vtu files.

    Originally from VTKDataFiles-9.3.0.tar.gz.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid | str
        Crushed can dataset or path depending on the value of ``load``.

    Examples
    --------
    Plot the crushed can dataset.

    >>> from pyvista import examples
    >>> import pyvista as pv
    >>> dataset = examples.download_can_crushed_vtu()
    >>> dataset.plot(smooth_shading=True)

    .. seealso::

        :ref:`Can Crushed Vtu Dataset <can_crushed_vtu_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Can Crushed Hdf Dataset <can_crushed_hdf_dataset>`

    """
    return _download_dataset(_dataset_can_crushed_vtu, load=load)


_dataset_can_crushed_vtu = _SingleFileDownloadableDatasetLoader('can.vtu')


@_deprecate_positional_args
def download_cgns_structured(load=True):  # noqa: FBT002
    """Download the structured CGNS dataset mesh.

    Originally downloaded from `CFD General Notation System Example Files
    <https://cgns.org/current/examples.html#constricting-channel>`_

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.MultiBlock | str
        Structured, 12 block, 3-D constricting channel, with example use of
        Family_t for BCs (ADF type). If ``load`` is ``False``, then the path of the
        example CGNS file is returned.

    Examples
    --------
    Plot the example CGNS dataset.

    >>> from pyvista import examples
    >>> import pyvista as pv
    >>> dataset = examples.download_cgns_structured()
    >>> dataset[0].plot(scalars='Density')

    .. seealso::

        :ref:`Cgns Structured Dataset <cgns_structured_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Cgns Multi Dataset <cgns_multi_dataset>`

    """
    return _download_dataset(_dataset_cgns_structured, load=load)


_dataset_cgns_structured = _SingleFileDownloadableDatasetLoader('cgns/sqnz_s.adf.cgns')


@_deprecate_positional_args
def download_tecplot_ascii(load=True):  # noqa: FBT002
    """Download the single block ASCII Tecplot dataset.

    Originally downloaded from Paul Bourke's
    `Sample file <http://paulbourke.net/dataformats/tp/sample.tp>`_

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.MultiBlock | str
        Multiblock format with only 1 data block, simple geometric shape.
        If ``load`` is ``False``, then the path of the example Tecplot file
        is returned.

    Examples
    --------
    Plot the example Tecplot dataset.

    >>> from pyvista import examples
    >>> import pyvista as pv
    >>> dataset = examples.download_tecplot_ascii()
    >>> dataset.plot()

    .. seealso::

        :ref:`Tecplot Ascii Dataset <tecplot_ascii_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_tecplot_ascii, load=load)


_dataset_tecplot_ascii = _SingleFileDownloadableDatasetLoader('tecplot_ascii.dat')


@_deprecate_positional_args
def download_cgns_multi(load=True):  # noqa: FBT002
    """Download a multielement airfoil with a cell centered solution.

    Originally downloaded from `CFD General Notation System Example Files
    <https://cgns.org/current/examples.html#d-multielement-airfoil>`_

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.MultiBlock | str
        Structured, 4 blocks, 2D (2 planes in third dimension) multielement
        airfoil, with cell centered solution. If ``load`` is ``False``, then the path of the
        example CGNS file is returned.

    Examples
    --------
    Plot the airfoil dataset. Merge the multi-block and then plot the airfoil's
    ``"ViscosityEddy"``. Convert the cell data to point data as in this
    dataset, the solution is stored within the cells.

    >>> from pyvista import examples
    >>> import pyvista as pv
    >>> dataset = examples.download_cgns_multi()
    >>> ugrid = dataset.combine()
    >>> ugrid = ugrid = ugrid.cell_data_to_point_data()
    >>> ugrid.plot(
    ...     cmap='bwr',
    ...     scalars='ViscosityEddy',
    ...     zoom=4,
    ...     cpos='xz',
    ...     show_scalar_bar=False,
    ... )

    .. seealso::

        :ref:`Cgns Multi Dataset <cgns_multi_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Cgns Structured Dataset <cgns_structured_dataset>`

    """
    return _download_dataset(_dataset_cgns_multi, load=load)


def _cgns_multi_read_func(filename):
    reader = pyvista.get_reader(filename)
    # Disable reading the boundary patch. This generates messages like
    # "Skipping BC_t node: BC_t type 'BCFarfield' not supported yet."
    reader.load_boundary_patch = False
    return reader.read()


_dataset_cgns_multi = _SingleFileDownloadableDatasetLoader(
    'cgns/multi.cgns',
    read_func=_cgns_multi_read_func,
)


@_deprecate_positional_args
def download_dicom_stack(
    load: bool = True,  # noqa: FBT001, FBT002
) -> pyvista.ImageData | str:
    """Download TCIA DICOM stack volume.

    Original download from the `The Cancer Imaging Archive (TCIA)
    <https://www.cancerimagingarchive.net/>`_. This is part of the
    Clinical Proteomic Tumor Analysis Consortium Sarcomas (CPTAC-SAR)
    collection.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ImageData | str
        DataSet or path depending on ``load``.

    References
    ----------
    * **Data Citation**

        National Cancer Institute Clinical Proteomic Tumor Analysis Consortium
        (CPTAC). (2018).  Radiology Data from the Clinical Proteomic Tumor
        Analysis Consortium Sarcomas [CPTAC-SAR] collection [Data set]. The
        Cancer Imaging Archive.  DOI: 10.7937/TCIA.2019.9bt23r95

    * **Acknowledgement**

        Data used in this publication were generated by the National Cancer Institute Clinical
        Proteomic Tumor Analysis Consortium (CPTAC).

    * **TCIA Citation**

        Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S,
        Maffitt D, Pringle M, Tarbox L, Prior F. The Cancer Imaging Archive (TCIA):
        Maintaining and Operating a Public Information Repository, Journal of Digital Imaging,
        Volume 26, Number 6, December, 2013, pp 1045-1057. doi: 10.1007/s10278-013-9622-7

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_dicom_stack()
    >>> dataset.plot(volume=True, zoom=3, show_scalar_bar=False)

    .. seealso::

        :ref:`Dicom Stack Dataset <dicom_stack_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`medical_dataset_gallery`
            Browse other medical datasets.

    """
    return _download_dataset(_dataset_dicom_stack, load=load)


_dataset_dicom_stack = _SingleFileDownloadableDatasetLoader(
    'DICOM_Stack/data.zip',
    target_file='data',
)


@_deprecate_positional_args
def download_parched_canal_4k(load=True):  # noqa: FBT002
    """Download parched canal 4k dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.Texture | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> texture = examples.download_parched_canal_4k()
    >>> texture.dimensions
    (4096, 2048)

    Use :meth:`~pyvista.ImageDataFilters.resample` to downsample the texture's
    underlying image before plotting.

    >>> _ = texture.to_image().resample(0.25, inplace=True)
    >>> texture.dimensions
    (1024, 512)

    >>> texture.plot(cpos='xy')

    .. seealso::

        :ref:`Parched Canal 4k Dataset <parched_canal_4k_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Dikhololo Night Dataset <dikhololo_night_dataset>`
            Another HDR texture.

    """
    return _download_dataset(_dataset_parched_canal_4k, load=load)


_dataset_parched_canal_4k = _SingleFileDownloadableDatasetLoader(
    'parched_canal_4k.hdr',
    read_func=read_texture,  # type: ignore[arg-type]
)


@_deprecate_positional_args
def download_cells_nd(load=True):  # noqa: FBT002
    """Download example AVS UCD dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_cells_nd()
    >>> dataset.plot(cpos='xy')

    .. seealso::

        :ref:`Cells Nd Dataset <cells_nd_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_cells_nd, load=load)


_dataset_cells_nd = _SingleFileDownloadableDatasetLoader('cellsnd.ascii.inp')


@_deprecate_positional_args
def download_moonlanding_image(load=True):  # noqa: FBT002
    """Download the Moon landing image.

    This is a noisy image originally obtained from `Scipy Lecture Notes
    <https://scipy-lectures.org/index.html>`_ and can be used to demonstrate a
    low pass filter.

    See the `scipy-lectures license
    <http://scipy-lectures.org/preface.html#license>`_ for more details
    regarding this image's use and distribution.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ImageData | str
        ``DataSet`` or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_moonlanding_image()
    >>> dataset.plot(
    ...     cpos='xy',
    ...     cmap='gray',
    ...     background='w',
    ...     show_scalar_bar=False,
    ... )

    .. seealso::

        :ref:`Moonlanding Image Dataset <moonlanding_image_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`image_fft_example`
            Full example using this dataset.

    """
    return _download_dataset(_dataset_moonlanding_image, load=load)


_dataset_moonlanding_image = _SingleFileDownloadableDatasetLoader('moonlanding.png')


@_deprecate_positional_args
def download_angular_sector(load=True):  # noqa: FBT002
    """Download the angular sector dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_angular_sector()
    >>> dataset.plot(scalars='PointId')

    .. seealso::

        :ref:`Angular Sector Dataset <angular_sector_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_angular_sector, load=load)


_dataset_angular_sector = _SingleFileDownloadableDatasetLoader('AngularSector.vtk')


@_deprecate_positional_args
def download_mount_damavand(load=True):  # noqa: FBT002
    """Download the Mount Damavand dataset.

    Visualize 3D models of Damavand Volcano, Alborz, Iran. This is a 2D map
    with the altitude embedded as ``'z'`` cell data within the
    :class:`pyvista.PolyData`.

    Originally posted at `banesullivan/damavand-volcano
    <https://github.com/banesullivan/damavand-volcano>`_.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Download the Damavand dataset and plot it after warping it by its altitude.

    >>> from pyvista import examples
    >>> dataset = examples.download_mount_damavand()
    >>> dataset = dataset.cell_data_to_point_data()
    >>> dataset = dataset.warp_by_scalar('z', factor=2)
    >>> dataset.plot(cmap='gist_earth', show_scalar_bar=False)

    .. seealso::

        :ref:`Mount Damavand Dataset <mount_damavand_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_mount_damavand, load=load)


_dataset_mount_damavand = _SingleFileDownloadableDatasetLoader('AOI.Damavand.32639.vtp')


@_deprecate_positional_args
def download_particles_lethe(load=True):  # noqa: FBT002
    """Download a particles dataset generated by `lethe <https://github.com/lethe-cfd/lethe>`_ .

    See `PyVista discussions #1984
    <https://github.com/pyvista/pyvista/discussions/1984>`_

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Download the particles dataset and plot it after generating glyphs.

    >>> from pyvista import examples
    >>> particles = examples.download_particles_lethe()
    >>> particles.plot(
    ...     render_points_as_spheres=True,
    ...     style='points',
    ...     scalars='Velocity',
    ...     background='w',
    ...     scalar_bar_args={'color': 'k'},
    ...     cmap='bwr',
    ... )

    .. seealso::

        :ref:`Particles Lethe Dataset <particles_lethe_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_particles_lethe, load=load)


_dataset_particles_lethe = _SingleFileDownloadableDatasetLoader(
    'lethe/result_particles.20000.0000.vtu',
)


@_deprecate_positional_args
def download_gif_simple(load=True):  # noqa: FBT002
    """Download a simple three frame GIF.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ImageData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Download and plot the first frame of a simple GIF.

    >>> from pyvista import examples
    >>> grid = examples.download_gif_simple()
    >>> grid.plot(
    ...     scalars='frame0',
    ...     rgb=True,
    ...     background='w',
    ...     show_scalar_bar=False,
    ...     cpos='xy',
    ... )

    Plot the second frame.

    >>> grid.plot(
    ...     scalars='frame1',
    ...     rgb=True,
    ...     background='w',
    ...     show_scalar_bar=False,
    ...     cpos='xy',
    ... )

    .. seealso::

        :ref:`Gif Simple Dataset <gif_simple_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_gif_simple, load=load)


_dataset_gif_simple = _SingleFileDownloadableDatasetLoader('gifs/sample.gif')


@_deprecate_positional_args
def download_cloud_dark_matter(load=True):  # noqa: FBT002
    """Download particles from a simulated dark matter halo.

    This dataset contains 32,314 particles.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PointSet | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Download the dark matter cloud and display its representation.

    >>> import numpy as np
    >>> from pyvista import examples
    >>> pc = examples.download_cloud_dark_matter()
    >>> pc
    PointSet (...)
      N Cells:    0
      N Points:   32314
      X Bounds:   7.451e+01, 7.892e+01
      Y Bounds:   1.616e+01, 2.275e+01
      Z Bounds:   8.900e+01, 9.319e+01
      N Arrays:   0

    Plot the point cloud. Color based on the distance from the center of the
    cloud.

    >>> pc.plot(
    ...     scalars=np.linalg.norm(pc.points - pc.center, axis=1),
    ...     style='points_gaussian',
    ...     opacity=0.5,
    ...     point_size=1.5,
    ...     show_scalar_bar=False,
    ...     zoom=2,
    ... )

    .. seealso::

        :ref:`Cloud Dark Matter Dataset <cloud_dark_matter_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Cloud Dark Matter Dense Dataset <cloud_dark_matter_dense_dataset>`

        :ref:`point_clouds_example`
            Full example using this dataset

    """
    return _download_dataset(_dataset_cloud_dark_matter, load=load)


_dataset_cloud_dark_matter = _SingleFileDownloadableDatasetLoader(
    'point-clouds/findus23/halo_low_res.npy',
    read_func=np.load,
    load_func=pyvista.PointSet,  # type: ignore[arg-type]
)


@_deprecate_positional_args
def download_cloud_dark_matter_dense(load=True):  # noqa: FBT002
    """Download a particles from a simulated dark matter halo.

    This dataset contains 2,062,256 particles.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PointSet | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Download the dark matter cloud and display its representation.

    >>> import numpy as np
    >>> from pyvista import examples
    >>> pc = examples.download_cloud_dark_matter_dense()
    >>> pc
    PointSet (...)
      N Cells:    0
      N Points:   2062256
      X Bounds:   7.462e+01, 7.863e+01
      Y Bounds:   1.604e+01, 2.244e+01
      Z Bounds:   8.893e+01, 9.337e+01
      N Arrays:   0

    Plot the point cloud. Color based on the distance from the center of the
    cloud.

    >>> pc.plot(
    ...     scalars=np.linalg.norm(pc.points - pc.center, axis=1),
    ...     style='points_gaussian',
    ...     opacity=0.030,
    ...     point_size=2.0,
    ...     show_scalar_bar=False,
    ...     zoom=2,
    ... )

    .. seealso::

        :ref:`Cloud Dark Matter Dense Dataset <cloud_dark_matter_dense_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Cloud Dark Matter Dataset <cloud_dark_matter_dataset>`

        :ref:`point_clouds_example`
            More details on how to plot point clouds.

    """
    return _download_dataset(_dataset_cloud_dark_matter_dense, load=load)


_dataset_cloud_dark_matter_dense = _SingleFileDownloadableDatasetLoader(
    'point-clouds/findus23/halo_high_res.npy',
    read_func=np.load,
    load_func=pyvista.PointSet,  # type: ignore[arg-type]
)


@_deprecate_positional_args
def download_stars_cloud_hyg(load=True):  # noqa: FBT002
    """Download a point cloud of stars as computed by the HYG Database.

    See `HYG-Database <https://github.com/astronexus/HYG-Database>`_ for more
    details.

    This data set is licensed by a Creative Commons Attribution-ShareAlike
    license. For more details, read the `Creative Commons page
    <https://creativecommons.org/licenses/by-sa/2.5/>`_

    See the `README.md
    <https://github.com/pyvista/vtk-data/blob/master/Data/point-clouds/hyg-database/README.md>`_
    for more details for how the star colors were computed.

    Distances are in parsecs from Earth.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Download and plot a point cloud of stars within 3,000 light years. Stars
    are colored according to their RGBA colors.

    >>> import numpy as np
    >>> from pyvista import examples
    >>> stars = examples.download_stars_cloud_hyg()
    >>> stars.plot(
    ...     style='points_gaussian',
    ...     background='k',
    ...     point_size=0.5,
    ...     scalars='_rgba',
    ...     render_points_as_spheres=False,
    ...     zoom=3.0,
    ... )

    >>> stars
    PolyData (...)
      N Cells:    107857
      N Points:   107857
      N Strips:   0
      X Bounds:   -9.755e+02, 9.774e+02
      Y Bounds:   -9.620e+02, 9.662e+02
      Z Bounds:   -9.788e+02, 9.702e+02
      N Arrays:   3

    .. seealso::

        :ref:`Stars Cloud Hyg Dataset <stars_cloud_hyg_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`point_clouds_example`
            More details on how to plot point clouds.

    """
    return _download_dataset(_dataset_stars_cloud_hyg, load=load)


_dataset_stars_cloud_hyg = _SingleFileDownloadableDatasetLoader(
    'point-clouds/hyg-database/stars.vtp',
)


@_deprecate_positional_args
def download_fea_bracket(load=True):  # noqa: FBT002
    """Download the finite element solution of a bracket.

    Contains von-mises equivalent cell stress assuming a vertical (y-axis) load.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Download and plot equivalent cell stress.

    >>> from pyvista import examples
    >>> grid = examples.download_fea_bracket()
    >>> grid.plot()

    Plot the point stress using the ``'jet'`` color map. Convert the cell data
    to point data.

    >>> from pyvista import examples
    >>> grid = examples.download_fea_bracket()
    >>> grid = grid.cell_data_to_point_data()
    >>> grid.plot(smooth_shading=True, split_sharp_edges=True, cmap='jet')

    .. seealso::

        :ref:`Fea Bracket Dataset <fea_bracket_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Fea Hertzian Contact Cylinder Dataset <fea_hertzian_contact_cylinder_dataset>`

        :ref:`Aero Bracket Dataset <aero_bracket_dataset>`

        :ref:`Notch Stress Dataset <notch_stress_dataset>`

        :ref:`Notch Displacement Dataset <notch_displacement_dataset>`

    """
    return _download_dataset(_dataset_fea_bracket, load=load)


_dataset_fea_bracket = _SingleFileDownloadableDatasetLoader('fea/kiefer/dataset.vtu')


@_deprecate_positional_args
def download_fea_hertzian_contact_cylinder(load=True):  # noqa: FBT002
    """Download a hertzian contact finite element solution.

    Hertzian contact is referred to the frictionless contact between two
    bodies. Spherical contact is a special case of the Hertz contact, which is
    between two spheres, or as in the case of this dataset, between a sphere
    and the surface of a half space (flat plane).

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Plot by part ID.

    >>> import numpy as np
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> grid = examples.download_fea_hertzian_contact_cylinder()
    >>> grid.plot(scalars='PartID', cmap=['green', 'blue'], show_scalar_bar=False)

    Plot the absolute value of the component stress in the Z direction.

    >>> pl = pv.Plotter()
    >>> z_stress = np.abs(grid['Stress'][:, 2])
    >>> _ = pl.add_mesh(
    ...     grid,
    ...     scalars=z_stress,
    ...     clim=[0, 1.2e9],
    ...     cmap='jet',
    ...     lighting=True,
    ...     show_edges=False,
    ...     ambient=0.2,
    ... )
    >>> pl.camera_position = 'xz'
    >>> pl.camera.zoom(1.4)
    >>> pl.show()

    .. seealso::

        :ref:`Fea Hertzian Contact Cylinder Dataset <fea_hertzian_contact_cylinder_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`fea_hertzian_contact_pressure_example`

        :ref:`Fea Bracket Dataset <fea_bracket_dataset>`

        :ref:`Aero Bracket Dataset <aero_bracket_dataset>`

        :ref:`Notch Stress Dataset <notch_stress_dataset>`

        :ref:`Notch Displacement Dataset <notch_displacement_dataset>`


    """
    return _download_dataset(_dataset_fea_hertzian_contact_cylinder, load=load)


_dataset_fea_hertzian_contact_cylinder = _SingleFileDownloadableDatasetLoader(
    'fea/hertzian_contact_cylinder/Hertzian_cylinder_on_plate.zip',
    target_file='bfac9fd1-e982-4825-9a95-9e5d8c5b4d3e_result_1.pvtu',
)


@_deprecate_positional_args
def download_black_vase(load=True, *, high_resolution=False):  # noqa: FBT002
    """Download a black vase scan created by Ivan Nikolov.

    The dataset was downloaded from `GGG-BenchmarkSfM: Dataset for Benchmarking
    Close-range SfM Software Performance under Varying Capturing Conditions
    <https://data.mendeley.com/datasets/bzxk2n78s9/4>`_

    Original datasets are under the CC BY 4.0 license.

    For more details, see `Ivan Nikolov Datasets
    <https://github.com/pyvista/vtk-data/tree/master/Data/ivan-nikolov>`_

    .. versionchanged:: 0.45

        A decimated version of this dataset with 31 thousand cells is now returned.
        Previously, a high-resolution version with 3.1 million cells was returned.
        Use ``high_resolution=True`` for the high-resolution version.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    high_resolution : bool, default: False
        Set this to ``True`` to return a high-resolution version of this dataset.
        By default, a :meth:`decimated <pyvista.PolyDataFilters.decimate>` version
        is returned with 99% reduction.

        .. versionadded:: 0.45

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Download and plot the dataset.

    >>> from pyvista import examples
    >>> mesh = examples.download_black_vase()
    >>> mesh.plot()

    Return the statistics of the dataset.

    >>> mesh
    PolyData (...)
      N Cells:    31366
      N Points:   17337
      N Strips:   0
      X Bounds:   -1.091e+02, 1.533e+02
      Y Bounds:   -1.200e+02, 1.416e+02
      Z Bounds:   1.667e+01, 4.078e+02
      N Arrays:   0

    .. seealso::

        :ref:`Black Vase Dataset <black_vase_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    if high_resolution:
        return _download_dataset(__dataset_black_vase_high_res, load=load)
    return _download_dataset(_dataset_black_vase, load=load)


_dataset_black_vase = _SingleFileDownloadableDatasetLoader(
    'ivan-nikolov/blackVase_decimated.vtp',
)
__dataset_black_vase_high_res = _SingleFileDownloadableDatasetLoader(
    'ivan-nikolov/blackVase.zip',
    target_file='blackVase.vtp',
)


@_deprecate_positional_args
def download_ivan_angel(load=True, *, high_resolution=False):  # noqa: FBT002
    """Download a scan of an angel statue created by Ivan Nikolov.

    The dataset was downloaded from `GGG-BenchmarkSfM: Dataset for Benchmarking
    Close-range SfM Software Performance under Varying Capturing Conditions
    <https://data.mendeley.com/datasets/bzxk2n78s9/4>`_

    Original datasets are under the CC BY 4.0 license.

    For more details, see `Ivan Nikolov Datasets
    <https://github.com/pyvista/vtk-data/tree/master/Data/ivan-nikolov>`_

    .. versionchanged:: 0.45

        A decimated version of this dataset with 36 thousand cells is now returned.
        Previously, a high-resolution version with 3.6 million cells was returned.
        Use ``high_resolution=True`` for the high-resolution version.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    high_resolution : bool, default: False
        Set this to ``True`` to return a high-resolution version of this dataset.
        By default, a :meth:`decimated <pyvista.PolyDataFilters.decimate>` version
        is returned with 99% reduction.

        .. versionadded:: 0.45

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Download and plot the dataset.

    >>> from pyvista import examples
    >>> mesh = examples.download_ivan_angel()
    >>> cpos = [
    ...     (-476.14, -393.73, 282.14),
    ...     (-15.00, 11.25, 44.08),
    ...     (0.26, 0.24, 0.93),
    ... ]
    >>> mesh.plot(cpos=cpos)

    Return the statistics of the dataset.

    >>> mesh
    PolyData (...)
      N Cells:    35804
      N Points:   18412
      N Strips:   0
      X Bounds:   -1.146e+02, 8.470e+01
      Y Bounds:   -6.987e+01, 9.254e+01
      Z Bounds:   -1.166e+02, 2.052e+02
      N Arrays:   0

    .. seealso::

        :ref:`Ivan Angel Dataset <ivan_angel_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    if high_resolution:
        return _download_dataset(__dataset_ivan_angel_high_res, load=load)
    return _download_dataset(_dataset_ivan_angel, load=load)


_dataset_ivan_angel = _SingleFileDownloadableDatasetLoader(
    'ivan-nikolov/Angel_decimated.vtp',
)
__dataset_ivan_angel_high_res = _SingleFileDownloadableDatasetLoader(
    'ivan-nikolov/Angel.zip',
    target_file='Angel.vtp',
)


@_deprecate_positional_args
def download_bird_bath(load=True, *, high_resolution=False):  # noqa: FBT002
    """Download a scan of a bird bath created by Ivan Nikolov.

    The dataset was downloaded from `GGG-BenchmarkSfM: Dataset for Benchmarking
    Close-range SfM Software Performance under Varying Capturing Conditions
    <https://data.mendeley.com/datasets/bzxk2n78s9/4>`_

    Original datasets are under the CC BY 4.0 license.

    For more details, see `Ivan Nikolov Datasets
    <https://github.com/pyvista/vtk-data/tree/master/Data/ivan-nikolov>`_

    .. versionchanged:: 0.45

        A decimated version of this dataset with 35 thousand cells is now returned.
        Previously, a high-resolution version with 3.5 million cells was returned.
        Use ``high_resolution=True`` for the high-resolution version.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    high_resolution : bool, default: False
        Set this to ``True`` to return a high-resolution version of this dataset.
        By default, a :meth:`decimated <pyvista.PolyDataFilters.decimate>` version
        is returned with 99% reduction.

        .. versionadded:: 0.45

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Download and plot the dataset.

    >>> from pyvista import examples
    >>> mesh = examples.download_bird_bath()
    >>> mesh.plot()

    Return the statistics of the dataset.

    >>> mesh
    PolyData (...)
      N Cells:    35079
      N Points:   18796
      N Strips:   0
      X Bounds:   -1.600e+02, 1.482e+02
      Y Bounds:   -1.522e+02, 1.547e+02
      Z Bounds:   -5.491e-01, 1.408e+02
      N Arrays:   0

    .. seealso::

        :ref:`Bird Bath Dataset <bird_bath_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    if high_resolution:
        return _download_dataset(__dataset_bird_bath_high_res, load=load)
    return _download_dataset(_dataset_bird_bath, load=load)


_dataset_bird_bath = _SingleFileDownloadableDatasetLoader('ivan-nikolov/birdBath_decimated.vtp')
__dataset_bird_bath_high_res = _SingleFileDownloadableDatasetLoader(
    'ivan-nikolov/birdBath.zip',
    target_file='birdBath.vtp',
)


@_deprecate_positional_args
def download_owl(load=True, *, high_resolution=False):  # noqa: FBT002
    """Download a scan of an owl statue created by Ivan Nikolov.

    The dataset was downloaded from `GGG-BenchmarkSfM: Dataset for Benchmarking
    Close-range SfM Software Performance under Varying Capturing Conditions
    <https://data.mendeley.com/datasets/bzxk2n78s9/4>`_

    Original datasets are under the CC BY 4.0 license.

    For more details, see `Ivan Nikolov Datasets
    <https://github.com/pyvista/vtk-data/tree/master/Data/ivan-nikolov>`_

    .. versionchanged:: 0.45

        A decimated version of this dataset with 24 thousand cells is now returned.
        Previously, a high-resolution version with 2.4 million cells was returned.
        Use ``high_resolution=True`` for the high-resolution version.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    high_resolution : bool, default: False
        Set this to ``True`` to return a high-resolution version of this dataset.
        By default, a :meth:`decimated <pyvista.PolyDataFilters.decimate>` version
        is returned with 99% reduction.

        .. versionadded:: 0.45

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Download and plot the dataset.

    >>> from pyvista import examples
    >>> mesh = examples.download_owl()
    >>> cpos = [
    ...     (-315.18, -402.21, 230.71),
    ...     (6.06, -1.74, 101.48),
    ...     (0.108, 0.226, 0.968),
    ... ]
    >>> mesh.plot(cpos=cpos)

    Return the statistics of the dataset.

    >>> mesh
    PolyData (...)
      N Cells:    24407
      N Points:   12442
      N Strips:   0
      X Bounds:   -5.834e+01, 7.048e+01
      Y Bounds:   -7.005e+01, 6.657e+01
      Z Bounds:   1.814e+00, 2.013e+02
      N Arrays:   0

    .. seealso::

        :ref:`Owl Dataset <owl_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    if high_resolution:
        return _download_dataset(__dataset_owl_high_res, load=load)
    return _download_dataset(_dataset_owl, load=load)


_dataset_owl = _SingleFileDownloadableDatasetLoader('ivan-nikolov/owl_decimated.vtp')
__dataset_owl_high_res = _SingleFileDownloadableDatasetLoader(
    'ivan-nikolov/owl.zip', target_file='owl.vtp'
)


@_deprecate_positional_args
def download_plastic_vase(load=True, *, high_resolution=False):  # noqa: FBT002
    """Download a scan of a plastic vase created by Ivan Nikolov.

    The dataset was downloaded from `GGG-BenchmarkSfM: Dataset for Benchmarking
    Close-range SfM Software Performance under Varying Capturing Conditions
    <https://data.mendeley.com/datasets/bzxk2n78s9/4>`_

    Original datasets are under the CC BY 4.0 license.

    For more details, see `Ivan Nikolov Datasets
    <https://github.com/pyvista/vtk-data/tree/master/Data/ivan-nikolov>`_

    .. versionchanged:: 0.45

        A decimated version of this dataset with 36 thousand cells is now returned.
        Previously, a high-resolution version with 3.6 million cells was returned.
        Use ``high_resolution=True`` for the high-resolution version.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    high_resolution : bool, default: False
        Set this to ``True`` to return a high-resolution version of this dataset.
        By default, a :meth:`decimated <pyvista.PolyDataFilters.decimate>` version
        is returned with 99% reduction.

        .. versionadded:: 0.45

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Download and plot the dataset.

    >>> from pyvista import examples
    >>> mesh = examples.download_plastic_vase()
    >>> mesh.plot()

    Return the statistics of the dataset.

    >>> mesh
    PolyData (...)
      N Cells:    35708
      N Points:   18238
      N Strips:   0
      X Bounds:   -1.364e+02, 1.928e+02
      Y Bounds:   -1.677e+02, 1.602e+02
      Z Bounds:   1.209e+02, 4.090e+02
      N Arrays:   0

    .. seealso::

        :ref:`Plastic Vase Dataset <plastic_vase_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    if high_resolution:
        return _download_dataset(__dataset_plastic_vase_high_res, load=load)
    return _download_dataset(_dataset_plastic_vase, load=load)


_dataset_plastic_vase = _SingleFileDownloadableDatasetLoader(
    'ivan-nikolov/plasticVase_decimated.vtp'
)
__dataset_plastic_vase_high_res = _SingleFileDownloadableDatasetLoader(
    'ivan-nikolov/plasticVase.zip',
    target_file='plasticVase.vtp',
)


@_deprecate_positional_args
def download_sea_vase(load=True, *, high_resolution=False):  # noqa: FBT002
    """Download a scan of a sea vase created by Ivan Nikolov.

    The dataset was downloaded from `GGG-BenchmarkSfM: Dataset for Benchmarking
    Close-range SfM Software Performance under Varying Capturing Conditions
    <https://data.mendeley.com/datasets/bzxk2n78s9/4>`_

    Original datasets are under the CC BY 4.0 license.

    For more details, see `Ivan Nikolov Datasets
    <https://github.com/pyvista/vtk-data/tree/master/Data/ivan-nikolov>`_

    .. versionchanged:: 0.45

        A decimated version of this dataset with 35 thousand cells is now returned.
        Previously, a high-resolution version with 3.5 million cells was returned.
        Use ``high_resolution=True`` for the high-resolution version.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    high_resolution : bool, default: False
        Set this to ``True`` to return a high-resolution version of this dataset.
        By default, a :meth:`decimated <pyvista.PolyDataFilters.decimate>` version
        is returned with 99% reduction.

        .. versionadded:: 0.45

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Download and plot the dataset.

    >>> from pyvista import examples
    >>> mesh = examples.download_sea_vase()
    >>> mesh.plot()

    Return the statistics of the dataset.

    >>> mesh
    PolyData (...)
      N Cells:    35483
      N Points:   18063
      N Strips:   0
      X Bounds:   -1.664e+02, 1.463e+02
      Y Bounds:   -1.741e+02, 1.382e+02
      Z Bounds:   -1.497e+02, 2.992e+02
      N Arrays:   0

    .. seealso::

        :ref:`Sea Vase Dataset <sea_vase_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    if high_resolution:
        return _download_dataset(__dataset_sea_vase_high_res, load=load)
    return _download_dataset(_dataset_sea_vase, load=load)


_dataset_sea_vase = _SingleFileDownloadableDatasetLoader('ivan-nikolov/seaVase_decimated.vtp')
__dataset_sea_vase_high_res = _SingleFileDownloadableDatasetLoader(
    'ivan-nikolov/seaVase.zip',
    target_file='seaVase.vtp',
)


@_deprecate_positional_args
def download_dikhololo_night(load=True):  # noqa: FBT002
    """Download and read the dikholo night hdr texture example.

    Files hosted at https://polyhaven.com/

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.Texture
        HDR Texture.

    Examples
    --------
    >>> from pyvista import examples
    >>> texture = examples.download_dikhololo_night()
    >>> texture.plot()

    .. seealso::

        :ref:`Dikhololo Night Dataset <dikhololo_night_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Parched Canal 4k Dataset <parched_canal_4k_dataset>`
            Another HDR texture.

        :ref:`load_gltf_example`
            See additional examples using this dataset.

    """
    return _download_dataset(_dataset_dikhololo_night, load=load)


def _dikhololo_night_load_func(texture):
    texture.SetColorModeToDirectScalars()
    texture.SetMipmap(True)
    texture.SetInterpolate(True)
    return texture


_dataset_dikhololo_night = _SingleFileDownloadableDatasetLoader(
    'dikhololo_night_4k.hdr',
    read_func=read_texture,  # type: ignore[arg-type]
)


@_deprecate_positional_args
def download_cad_model_case(load=True):  # noqa: FBT002
    """Download a CAD model of a Raspberry PI 4 case.

    The dataset was downloaded from `Thingiverse
    <https://www.thingiverse.com/thing:4947746>`_

    Original datasets are under the `Creative Commons - Attribution
    <https://creativecommons.org/licenses/by/4.0/>`_ license.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Download and plot the dataset.

    >>> from pyvista import examples
    >>> mesh = examples.download_cad_model_case()
    >>> mesh.plot()

    Return the statistics of the dataset.

    >>> mesh
    PolyData (...)
      N Cells:    15446
      N Points:   7677
      N Strips:   0
      X Bounds:   -6.460e-31, 9.000e+01
      Y Bounds:   -3.535e-32, 1.480e+02
      Z Bounds:   0.000e+00, 2.000e+01
      N Arrays:   2

    .. seealso::

        :ref:`Cad Model Case Dataset <cad_model_case_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_cad_model_case, load=load)


_dataset_cad_model_case = _SingleFileDownloadableDatasetLoader(
    'cad/4947746/Vented_Rear_Case_With_Pi_Supports.vtp',
)


@_deprecate_positional_args
def download_aero_bracket(load=True):  # noqa: FBT002
    """Download the finite element solution of an aero bracket.

    Data generated from public SimScale examples at `SimScale Project Library -
    Turbo <https://www.simscale.com/projects/ayarnoz/turbo/>`_.

    Licensing for this dataset is granted to freely and without restriction
    reproduce, distribute, publish according to the `SimScale Terms and
    Conditions <https://www.simscale.com/terms-and-conditions/>`_.

    This project demonstrates the static stress analysis of three aircraft
    engine bearing bracket models considering both linear and nonlinear
    material definition. The models are tested with horizontal and vertical
    loading conditions as provided on the `GrabCAD - Airplane Bearing Bracket
    Challenge
    <https://grabcad.com/challenges/airplane-bearing-bracket-challenge/entries>`_.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Download the aero bracket.

    >>> from pyvista import examples
    >>> dataset = examples.download_aero_bracket()
    >>> dataset
    UnstructuredGrid (...)
      N Cells:    117292
      N Points:   187037
      X Bounds:   -6.858e-03, 1.118e-01
      Y Bounds:   -1.237e-02, 6.634e-02
      Z Bounds:   -1.638e-02, 1.638e-02
      N Arrays:   3

    Show the available point data arrays.

    >>> dataset.point_data
    pyvista DataSetAttributes
    Association     : POINT
    Active Scalars  : None
    Active Vectors  : None
    Active Texture  : None
    Active Normals  : None
    Contains arrays :
        displacement            float32    (187037, 3)
        total nonlinear strain  float32    (187037, 6)
        von Mises stress        float32    (187037,)

    Plot the von Mises stress.

    >>> cpos = [
    ...     (-0.0503, 0.132, -0.179),
    ...     (0.0505, 0.0185, -0.00201),
    ...     (0.275, 0.872, 0.405),
    ... ]
    >>> dataset.plot(
    ...     smooth_shading=True,
    ...     split_sharp_edges=True,
    ...     scalars='von Mises stress',
    ...     cmap='bwr',
    ...     cpos=cpos,
    ...     anti_aliasing='fxaa',
    ... )

    .. seealso::

        :ref:`Aero Bracket Dataset <aero_bracket_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Notch Stress Dataset <notch_stress_dataset>`

        :ref:`Notch Displacement Dataset <notch_displacement_dataset>`

        :ref:`Fea Bracket Dataset <fea_bracket_dataset>`

        :ref:`Fea Hertzian Contact Cylinder Dataset <fea_hertzian_contact_cylinder_dataset>`

    """
    return _download_dataset(_dataset_aero_bracket, load=load)


_dataset_aero_bracket = _SingleFileDownloadableDatasetLoader('fea/aero_bracket/aero_bracket.vtu')


@_deprecate_positional_args
def download_coil_magnetic_field(load=True):  # noqa: FBT002
    """Download the magnetic field of a coil.

    These examples were generated from the following `script
    <https://github.com/pyvista/vtk-data/tree/master/Data/magpylib/>`_.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ImageData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Download the magnetic field dataset and generate streamlines from the field.

    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> grid = examples.download_coil_magnetic_field()
    >>> seed = pv.Disc(inner=1, outer=5.2, r_res=3, c_res=12)
    >>> strl = grid.streamlines_from_source(
    ...     seed,
    ...     vectors='B',
    ...     max_length=180,
    ...     initial_step_length=0.1,
    ...     integration_direction='both',
    ... )
    >>> strl.plot(
    ...     cmap='plasma',
    ...     render_lines_as_tubes=True,
    ...     line_width=2,
    ...     lighting=False,
    ...     zoom=2,
    ... )

    Plot the magnet field strength in the Z direction.

    >>> import numpy as np
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> grid = examples.download_coil_magnetic_field()
    >>> # create coils
    >>> coils = []
    >>> for z in np.linspace(-8, 8, 16):
    ...     coils.append(
    ...         pv.Polygon(center=(0, 0, z), radius=5, n_sides=100, fill=False)
    ...     )
    >>> coils = pv.MultiBlock(coils)
    >>> # plot the magnet field strength in the Z direction
    >>> scalars = np.abs(grid['B'][:, 2])
    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(coils, render_lines_as_tubes=True, line_width=5, color='w')
    >>> vol = pl.add_volume(
    ...     grid,
    ...     scalars=scalars,
    ...     cmap='plasma',
    ...     show_scalar_bar=False,
    ...     log_scale=True,
    ...     opacity='sigmoid_2',
    ... )
    >>> vol.prop.interpolation_type = 'linear'
    >>> _ = pl.add_volume_clip_plane(
    ...     vol,
    ...     normal='-x',
    ...     normal_rotation=False,
    ...     interaction_event='always',
    ...     widget_color=pv.Color(opacity=0.0),
    ... )
    >>> pl.enable_anti_aliasing()
    >>> pl.camera.zoom(2)
    >>> pl.show()

    .. seealso::

        :ref:`Coil Magnetic Field Dataset <coil_magnetic_field_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`magnetic_fields_example`
            More details on how to plot with this dataset.

    """
    return _download_dataset(_dataset_coil_magnetic_field, load=load)


_dataset_coil_magnetic_field = _SingleFileDownloadableDatasetLoader('magpylib/coil_field.vti')


@_deprecate_positional_args
def download_meshio_xdmf(load=True):  # noqa: FBT002
    """Download xdmf file created by meshio.

    The dataset was created by ``test_time_series`` test function in meshio.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_meshio_xdmf()
    >>> dataset.plot()

    .. seealso::

        :ref:`Meshio Xdmf Dataset <meshio_xdmf_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_meshio_xdmf, load=load)


def _meshio_xdmf_files_func():
    h5 = _DownloadableFile('meshio/out.h5')
    xdmf = _SingleFileDownloadableDatasetLoader('meshio/out.xdmf')
    return xdmf, h5


_dataset_meshio_xdmf = _MultiFileDownloadableDatasetLoader(files_func=_meshio_xdmf_files_func)


@_deprecate_positional_args
def download_victorian_goblet_face_illusion(load=True):  # noqa: FBT002
    """Download Victorian Goblet face illusion.

    This is a replica of a Victorian goblet with an external profile
    which resembles that of a face.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> import pyvista as pv
    >>> mesh = examples.download_victorian_goblet_face_illusion()
    >>> plotter = pv.Plotter(lighting='none')
    >>> _ = plotter.add_mesh(
    ...     mesh, edge_color='gray', color='white', show_edges=True
    ... )
    >>> _ = plotter.add_floor('-x', color='black')
    >>> plotter.enable_parallel_projection()
    >>> plotter.show(cpos='yz')

    .. seealso::

        :ref:`Victorian Goblet Face Illusion Dataset <victorian_goblet_face_illusion_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_victorian_goblet_face_illusion, load=load)


_dataset_victorian_goblet_face_illusion = _SingleFileDownloadableDatasetLoader(
    'Victorian_Goblet_face_illusion/Vase.stl',
)


@_deprecate_positional_args
def download_reservoir(load=True):  # noqa: FBT002
    """Download the UNISIM-II-D reservoir model.

    UNISIM-II is a synthetic carbonate reservoir model created by
    UNISIM-CEPETRO-Unicamp. The dataset can be used to compare methodologies
    and performance of different techniques, simulators, algorithms, among others.
    See more at https://www.unisim.cepetro.unicamp.br/benchmarks/br/unisim-ii/overview

    This dataset is licenced under the Database Contents License: http://opendatacommons.org/licenses/dbcl/1.0/

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ExplicitStructuredGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Load and plot dataset.

    >>> from pyvista import examples
    >>> import pyvista as pv
    >>> dataset = examples.download_reservoir()
    >>> dataset
    ExplicitStructuredGrid (...)
      N Cells:    47610
      N Points:   58433
      X Bounds:   3.104e+05, 3.177e+05
      Y Bounds:   7.477e+06, 7.486e+06
      Z Bounds:   -2.472e+03, -1.577e+03
      N Arrays:   6


    >>> plot = pv.Plotter()
    >>> _ = plot.add_mesh(dataset, show_edges=True)
    >>> camera = plot.camera
    >>> camera.position = (312452, 7474760, 3507)
    >>> camera.focal_point = (314388, 7481520, -2287)
    >>> camera.up = (0.09, 0.63, 0.77)
    >>> camera.distance = 9112
    >>> camera.clipping_range = (595, 19595)
    >>> plot.show()

    .. seealso::

        :ref:`Reservoir Dataset <reservoir_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_reservoir, load=load)


def _reservoir_load_func(grid):
    # See loading steps from this example:
    # https://examples.vtk.org/site/Python/ExplicitStructuredGrid/LoadESGrid/
    grid.ComputeFacesConnectivityFlagsArray()
    grid.set_active_scalars('ConnectivityFlags')

    # Remove misc data fields stored with the dataset
    grid.field_data.remove('dimensions')
    grid.field_data.remove('name')
    grid.field_data.remove('properties')
    grid.field_data.remove('filename')

    return grid


_dataset_reservoir = _SingleFileDownloadableDatasetLoader(
    'reservoir/UNISIM-II-D.zip',
    target_file='UNISIM-II-D.vtu',
    read_func=pyvista.ExplicitStructuredGrid,  # type: ignore[arg-type]
    load_func=_reservoir_load_func,
)


@_deprecate_positional_args
def download_whole_body_ct_male(
    load=True,  # noqa: FBT002
    *,
    high_resolution=False,
):
    r"""Download a CT image of a male subject with 117 segmented anatomic structures.

    This dataset is subject ``'s1397'`` from the TotalSegmentator dataset, version 2.0.1,
    available from `zenodo <https://zenodo.org/records/10047292>`_. See the
    original paper for details:

    Jakob Wasserthal et al., “TotalSegmentator: Robust Segmentation of 104 Anatomic
    Structures in CT Images,” Radiology, Jul. 2023, doi: https://doi.org/10.1148/ryai.230024.

    The dataset is loaded as a :class:`~pyvista.MultiBlock` with three blocks:

    -   ``'ct'``: :class:`~pyvista.ImageData` with CT data.

    -   ``'segmentations'``: :class:`~pyvista.MultiBlock` with 117 :class:`~pyvista.ImageData`
        blocks, each containing a binary segmentation label. The blocks are named by
        their anatomic structure (e.g. ``'heart'``) and are sorted alphabetically. See the
        examples below for a complete list label names.

    -   ``'label_map'``: :class:`~pyvista.ImageData` with a label map array. The
        label map is an alternative representation of the segmentation where
        the masks are combined into a single scalar array.

        .. note::

            The label map is not part of the original data source.

    Licensed under Creative Commons Attribution 4.0 International.

    .. versionadded:: 0.45

        Three dictionaries are now included with the dataset's
        :class:`~pyvista.DataObject.user_dict` to map label names to ids and
        colors:

        - ``'names_to_colors'`` : maps segment names to 8-bit RGB colors.
        - ``'names_to_ids'`` : maps segment names to integer ids used by the label map.
        - ``'ids_to_colors'`` : maps label ids to colors.

        The label ids are the ids used by the included label map.

    .. versionchanged:: 0.45

        A downsampled version of this dataset with dimensions ``(160, 160, 273)``
        is now returned. Previously, a high-resolution version with dimensions
        ``(320, 320, 547)`` was returned. Use ``high_resolution=True`` for the
        high-resolution version.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    high_resolution : bool, default: False
        Set this to ``True`` to return a high-resolution version of this dataset.
        By default, a :meth:`resampled <pyvista.ImageDataFilters.resample>` version
        with a ``0.5`` sampling rate is returned.

        .. versionadded:: 0.45

    Returns
    -------
    pyvista.MultiBlock or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Load the dataset and get some of its properties.

    >>> from pyvista import examples
    >>> import pyvista as pv
    >>> dataset = examples.download_whole_body_ct_male()

    Get the CT image.

    >>> ct_image = dataset['ct']
    >>> ct_image
    ImageData (...)
      N Cells:      6876432
      N Points:     6988800
      X Bounds:     7.500e-01, 4.778e+02
      Y Bounds:     7.500e-01, 4.778e+02
      Z Bounds:     7.527e-01, 8.182e+02
      Dimensions:   160, 160, 273
      Spacing:      3.000e+00, 3.000e+00, 3.005e+00
      N Arrays:     1

    Get the segmentation label names and show the first three.

    >>> segmentations = dataset['segmentations']
    >>> label_names = segmentations.keys()
    >>> label_names[:3]
    ['adrenal_gland_left', 'adrenal_gland_right', 'aorta']

    Get the label map and show its data range.

    >>> label_map = dataset['label_map']
    >>> label_map.get_data_range()
    (np.uint8(0), np.uint8(117))

    Show the ``'names_to_colors'`` dictionary with RGB colors for each segment.

    >>> dataset.user_dict['names_to_colors']  # doctest: +SKIP

    Show the ``'names_to_ids'`` dictionary with a mapping from segment names to segment ids.

    >>> dataset.user_dict['names_to_ids']  # doctest: +SKIP

    Create a surface mesh of the segmentation labels.

    >>> labels_mesh = label_map.contour_labels()

    Color the surface using :func:`~pyvista.DataSetFilters.color_labels`. Use the
    ``'ids_to_colors'`` dictionary that's included with the dataset to map the colors.

    >>> colored_mesh = labels_mesh.color_labels(
    ...     colors=dataset.user_dict['ids_to_colors']
    ... )

    Plot the CT image and segmentation labels together.

    >>> pl = pv.Plotter()
    >>> _ = pl.add_volume(
    ...     ct_image,
    ...     cmap='bone',
    ...     opacity='sigmoid_8',
    ...     show_scalar_bar=False,
    ... )
    >>> _ = pl.add_mesh(colored_mesh)
    >>> pl.view_zx()
    >>> pl.camera.up = (0, 0, 1)
    >>> pl.camera.zoom(1.3)
    >>> pl.show()

    .. seealso::

        :ref:`anatomical_groups_example`
            Additional examples using this dataset.

        :ref:`Whole Body Ct Male Dataset <whole_body_ct_male_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Whole Body Ct Female Dataset <whole_body_ct_female_dataset>`
            Similar dataset of a female subject.

        :ref:`medical_dataset_gallery`
            Browse other medical datasets.

        :ref:`crop_labeled_example`
            Example cropping this dataset using a segmentation mask.

        :ref:`volume_with_mask_example`
            See additional examples using this dataset.

    """
    if high_resolution:
        return _download_dataset(__dataset_whole_body_ct_male_high_res, load=load)
    return _download_dataset(_dataset_whole_body_ct_male, load=load)


class _WholeBodyCTUtilities:
    @staticmethod
    def import_colors_dict(module_path):
        # Import `colors` dict from downloaded `colors.py` module
        module_name = 'colors'
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is not None:
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)  # type:ignore[union-attr]
            from colors import colors  # noqa: PLC0415

            return dict(sorted(colors.items()))
        else:
            msg = 'Unable to load colors.'
            raise RuntimeError(msg)

    @staticmethod
    def add_metadata(dataset: pyvista.MultiBlock, colors_module_path: str):
        # Add color and id mappings to dataset
        segmentations = cast('pyvista.MultiBlock', dataset['segmentations'])
        label_names = sorted(segmentations.keys())
        names_to_colors = _WholeBodyCTUtilities.import_colors_dict(colors_module_path)
        names_to_ids = {key: i + 1 for i, key in enumerate(label_names)}
        dataset.user_dict['names_to_colors'] = names_to_colors
        dataset.user_dict['names_to_ids'] = names_to_ids
        dataset.user_dict['ids_to_colors'] = dict(
            sorted({names_to_ids[name]: names_to_colors[name] for name in label_names}.items())
        )

    @staticmethod
    def label_map_from_masks(masks: pyvista.MultiBlock):
        # Create label map array from segmentation masks
        # Initialize array with background values (zeros)
        n_points = cast('pyvista.ImageData', masks[0]).n_points
        label_map_array = np.zeros((n_points,), dtype=np.uint8)
        label_names = sorted(masks.keys())
        for i, name in enumerate(label_names):
            mask = cast('pyvista.ImageData', masks[name])
            label_map_array[mask.active_scalars == 1] = i + 1

        # Add scalars to a new image
        label_map_image = pyvista.ImageData()
        label_map_image.copy_structure(cast('pyvista.ImageData', masks[0]))
        label_map_image['label_map'] = label_map_array  # type: ignore[assignment]
        return label_map_image

    @staticmethod
    def load_func(files):
        dataset_file, colors_module = files
        dataset = dataset_file.load()

        # Create label map and add to dataset
        dataset['label_map'] = _WholeBodyCTUtilities.label_map_from_masks(dataset['segmentations'])

        # Add metadata
        _WholeBodyCTUtilities.add_metadata(dataset, colors_module.path)
        return dataset

    @staticmethod
    def files_func(name):
        # Resampled version is saved as a multiblock
        target_file = f'{name}.vtm' if 'resampled' in name else name

        def func():
            # Multiple files needed for read, but only one gets loaded
            dataset = _SingleFileDownloadableDatasetLoader(
                f'whole_body_ct/{name}.zip',
                target_file=target_file,
            )
            colors = _DownloadableFile('whole_body_ct/colors.py')
            return dataset, colors

        return func


_dataset_whole_body_ct_male = _MultiFileDownloadableDatasetLoader(
    _WholeBodyCTUtilities.files_func('s1397_resampled'),
    load_func=_WholeBodyCTUtilities.load_func,
)
__dataset_whole_body_ct_male_high_res = _MultiFileDownloadableDatasetLoader(
    _WholeBodyCTUtilities.files_func('s1397'), load_func=_WholeBodyCTUtilities.load_func
)


@_deprecate_positional_args
def download_whole_body_ct_female(
    load=True,  # noqa: FBT002
    *,
    high_resolution=False,
):
    r"""Download a CT image of a female subject with 117 segmented anatomic structures.

    This dataset is subject ``'s1380'`` from the TotalSegmentator dataset, version 2.0.1,
    available from `zenodo <https://zenodo.org/records/10047292>`_. See the
    original paper for details:

    Jakob Wasserthal et al., “TotalSegmentator: Robust Segmentation of 104 Anatomic
    Structures in CT Images,” Radiology, Jul. 2023, doi: https://doi.org/10.1148/ryai.230024.

    The dataset is loaded as a :class:`~pyvista.MultiBlock` with three blocks:

    -   ``'ct'``: :class:`~pyvista.ImageData` with CT data.

    -   ``'segmentations'``: :class:`~pyvista.MultiBlock` with 117 :class:`~pyvista.ImageData`
        blocks, each containing a binary segmentation label. The blocks are named by
        their anatomic structure (e.g. ``'heart'``) and are sorted alphabetically. See the
        examples below for a complete list label names.

    -   ``'label_map'``: :class:`~pyvista.ImageData` with a label map array. The
        label map is an alternative representation of the segmentation where
        the masks are combined into a single scalar array.

        .. note::

            The label map is not part of the original data source.

    Licensed under Creative Commons Attribution 4.0 International.

    .. versionadded:: 0.45

        Three dictionaries are now included with the dataset's
        :class:`~pyvista.DataObject.user_dict` to map label names to ids and
        colors:

        - ``'names_to_colors'`` : maps segment names to 8-bit RGB colors.
        - ``'names_to_ids'`` : maps segment names to integer ids used by the label map.
        - ``'ids_to_colors'`` : maps label ids to colors.

        The label ids are the ids used by the included label map.

    .. versionchanged:: 0.45

        A downsampled version of this dataset with dimensions ``(160, 160, 273)``
        is now returned. Previously, a high-resolution version with dimensions
        ``(320, 320, 547)`` was returned. Use ``high_resolution=True`` for the
        high-resolution version.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    high_resolution : bool, default: False
        Set this to ``True`` to return a high-resolution version of this dataset.
        By default, a :meth:`resampled <pyvista.ImageDataFilters.resample>` version
        with a ``0.5`` sampling rate is returned.

        .. versionadded:: 0.45

    Returns
    -------
    pyvista.MultiBlock or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Load the dataset.

    >>> from pyvista import examples
    >>> import pyvista as pv
    >>> dataset = examples.download_whole_body_ct_female()

    Get the names of the dataset's blocks.

    >>> dataset.keys()
    ['ct', 'segmentations', 'label_map']

    Get the CT image.

    >>> ct_image = dataset['ct']
    >>> ct_image
    ImageData (...)
      N Cells:      6825870
      N Points:     6937600
      X Bounds:     7.500e-01, 4.778e+02
      Y Bounds:     7.500e-01, 4.778e+02
      Z Bounds:     7.528e-01, 8.122e+02
      Dimensions:   160, 160, 271
      Spacing:      3.000e+00, 3.000e+00, 3.006e+00
      N Arrays:     1

    Get the segmentation label names and show the first three.

    >>> segmentations = dataset['segmentations']
    >>> label_names = segmentations.keys()
    >>> label_names[:3]
    ['adrenal_gland_left', 'adrenal_gland_right', 'aorta']

    Get the label map and show its data range.

    >>> label_map = dataset['label_map']
    >>> label_map.get_data_range()
    (np.uint8(0), np.uint8(117))

    Show the ``'names_to_colors'`` dictionary with RGB colors for each segment.

    >>> dataset.user_dict['names_to_colors']  # doctest: +SKIP

    Show the ``'names_to_ids'`` dictionary with a mapping from segment names to segment ids.

    >>> dataset.user_dict['names_to_ids']  # doctest: +SKIP

    Create a surface mesh of the segmentation labels.

    >>> labels_mesh = label_map.contour_labels()

    Color the surface using :func:`~pyvista.DataSetFilters.color_labels`. Use the
    ``'ids_to_colors'`` dictionary included with the dataset to map the colors.

    >>> colored_mesh = labels_mesh.color_labels(
    ...     colors=dataset.user_dict['ids_to_colors']
    ... )

    Plot the CT image and segmentation labels together.

    >>> pl = pv.Plotter()
    >>> _ = pl.add_volume(
    ...     ct_image,
    ...     cmap='bone',
    ...     opacity='sigmoid_7',
    ...     show_scalar_bar=False,
    ... )
    >>> _ = pl.add_mesh(colored_mesh)
    >>> pl.view_zx()
    >>> pl.camera.up = (0, 0, 1)
    >>> pl.camera.zoom(1.3)
    >>> pl.show()

    .. seealso::

        :ref:`anatomical_groups_example`
            Additional examples using this dataset.

        :ref:`Whole Body Ct Female Dataset <whole_body_ct_female_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Whole Body Ct Male Dataset <whole_body_ct_male_dataset>`
            Similar dataset of a male subject.

        :ref:`medical_dataset_gallery`
            Browse other medical datasets.

        :ref:`crop_labeled_example`
            Example cropping this dataset using a segmentation mask.

        :ref:`volume_with_mask_example`
            See additional examples using this dataset.

    """
    if high_resolution:
        return _download_dataset(__dataset_whole_body_ct_female_high_res, load=load)
    return _download_dataset(_dataset_whole_body_ct_female, load=load)


_dataset_whole_body_ct_female = _MultiFileDownloadableDatasetLoader(
    _WholeBodyCTUtilities.files_func('s1380_resampled'),
    load_func=_WholeBodyCTUtilities.load_func,
)
__dataset_whole_body_ct_female_high_res = _MultiFileDownloadableDatasetLoader(
    _WholeBodyCTUtilities.files_func('s1380'), load_func=_WholeBodyCTUtilities.load_func
)


@_deprecate_positional_args
def download_room_cff(load=True):  # noqa: FBT002
    """Download a room model in CFF format.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.MultiBlock or tuple
        DataSet or tuple of filenames depending on ``load``.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> blocks = examples.download_room_cff()
    >>> mesh = blocks[0]
    >>> mesh.plot(cpos='xy', scalars='SV_T')

    .. seealso::

        :ref:`Room Cff Dataset <room_cff_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_room_cff, load=load)


def _dataset_room_cff_files_func():
    cas = _SingleFileDownloadableDatasetLoader('FLUENTCFF/room.cas.h5')
    dat = _DownloadableFile('FLUENTCFF/room.dat.h5')
    return cas, dat


_dataset_room_cff = _MultiFileDownloadableDatasetLoader(_dataset_room_cff_files_func)


@_deprecate_positional_args
def download_m4_total_density(load=True):  # noqa: FBT002
    """Download a total density dataset of the chemistry.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ImageData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples

    >>> filename = examples.download_m4_total_density(load=False)
    >>> reader = pv.get_reader(filename)
    >>> reader.hb_scale = 1.1
    >>> reader.b_scale = 10.0

    >>> grid = reader.read()
    >>> poly = reader.read(grid=False)

    Add the outline and volume to the plotter.

    >>> pl = pv.Plotter()
    >>> outline = pl.add_mesh(grid.outline(), color='black')
    >>> volume = pl.add_volume(grid)

    Add atoms and bonds to the plotter.

    >>> atoms = pl.add_mesh(poly.glyph(geom=pv.Sphere()), color='red')
    >>> bonds = pl.add_mesh(poly.tube(), color='white')

    >>> pl.show(cpos='zx')

    .. seealso::

        :ref:`M4 Total Density Dataset <m4_total_density_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_m4_total_density, load=load)


_dataset_m4_total_density = _SingleFileDownloadableDatasetLoader('m4_TotalDensity.cube')


@_deprecate_positional_args
def download_headsq(load=True):  # noqa: FBT002
    """Download the headsq dataset.

    The headsq dataset is a 3D MRI scan of a human head.

    .. versionadded:: 0.44.0

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ImageData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> mesh = examples.download_headsq()
    >>> mesh.plot(cpos='xy')

    .. seealso::

        :ref:`Headsq Dataset <headsq_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_headsq, load=load)


def _dataset_headsq_files_func():
    return tuple(
        [_SingleFileDownloadableDatasetLoader('headsq/quarter.nhdr')]
        + [_DownloadableFile('headsq/quarter.' + str(i)) for i in range(1, 94)],
    )


_dataset_headsq = _MultiFileDownloadableDatasetLoader(_dataset_headsq_files_func)


@_deprecate_positional_args
def download_prism(load=True):  # noqa: FBT002
    """Download a prism model.

    .. versionadded:: 0.44.0

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> mesh = examples.download_prism()
    >>> mesh.plot()

    .. seealso::

        :ref:`Prism Dataset <prism_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_prism, load=load)


_dataset_prism = _SingleFileDownloadableDatasetLoader('prism.neu')


@_deprecate_positional_args
def download_t3_grid_0(load=True):  # noqa: FBT002
    """Download a T3 grid 0 image.

    .. versionadded:: 0.44.0

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ImageData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> mesh = examples.download_t3_grid_0()
    >>> mesh.plot()

    .. seealso::

        :ref:`T3 Grid 0 Dataset <t3_grid_0_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_t3_grid_0, load=load)


_dataset_t3_grid_0 = _SingleFileDownloadableDatasetLoader('t3_grid_0.mnc')


@_deprecate_positional_args
def download_caffeine(load=True):  # noqa: FBT002
    """Download the caffeine molecule.

    .. versionadded:: 0.44.0

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> filename = examples.download_caffeine(load=False)
    >>> reader = pv.get_reader(filename)
    >>> poly = reader.read()

    Add atoms and bonds to the plotter.

    >>> pl = pv.Plotter()
    >>> atoms = pl.add_mesh(poly.glyph(geom=pv.Sphere(radius=0.1)), color='red')
    >>> bonds = pl.add_mesh(poly.tube(radius=0.1), color='gray')
    >>> pl.show(cpos='xy')

    .. seealso::

        :ref:`Caffeine Dataset <caffeine_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_caffeine, load=load)


_dataset_caffeine = _SingleFileDownloadableDatasetLoader('caffeine.pdb')


@_deprecate_positional_args
def download_e07733s002i009(load=True):  # paragma: no cover  # noqa: FBT002
    """Download a e07733s002i009 image.

    .. versionadded:: 0.44.0

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ImageData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> mesh = examples.download_e07733s002i009()
    >>> mesh.plot()

    .. seealso::

        :ref:`E07733s002i009 Dataset <e07733s002i009_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_e07733s002i009, load=load)


_dataset_e07733s002i009 = _SingleFileDownloadableDatasetLoader('E07733S002I009.MR')


@_deprecate_positional_args
def download_particles(load=True):  # noqa: FBT002
    """Download a particle dataset.

    .. versionadded:: 0.44.0

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> filename = examples.download_particles(load=False)
    >>> reader = pv.get_reader(filename)
    >>> reader.reader.SetDataByteOrderToBigEndian()
    >>> reader.reader.Update()
    >>> mesh = reader.read()
    >>> mesh.plot()

    .. seealso::

        :ref:`Particles Dataset <particles_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_particles, load=load)


_dataset_particles = _SingleFileDownloadableDatasetLoader('Particles.raw')


@_deprecate_positional_args
def download_prostar(load=True):  # noqa: FBT002
    """Download a prostar dataset.

    .. versionadded:: 0.44.0

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> mesh = examples.download_prostar()
    >>> mesh.plot()

    .. seealso::

        :ref:`Prostar Dataset <prostar_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_prostar, load=load)


def _prostar_files_func():
    # Multiple files needed for read, but only one gets loaded
    prostar_cel = _DownloadableFile('prostar.cel')
    prostar_vrt = _SingleFileDownloadableDatasetLoader('prostar.vrt')
    return prostar_vrt, prostar_cel


_dataset_prostar = _MultiFileDownloadableDatasetLoader(_prostar_files_func)


@_deprecate_positional_args
def download_3gqp(load=True):  # noqa: FBT002
    """Download a 3GQP dataset.

    .. versionadded:: 0.44.0

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> mesh = examples.download_3gqp()
    >>> mesh.plot()

    .. seealso::

        :ref:`3gqp Dataset <3gqp_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_3gqp, load=load)


_dataset_3gqp = _SingleFileDownloadableDatasetLoader('3GQP.pdb')


@_deprecate_positional_args
def download_full_head(load=True):  # noqa: FBT002
    """Download the full head image.

    .. versionadded:: 0.45.0

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ImageData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_full_head()
    >>> dataset.plot(volume=True)

    .. seealso::

        :ref:`Full Head Dataset <full_head_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_full_head, load=load)


def _full_head_files_func():
    full_head_raw = _DownloadableFile('FullHead.raw.gz')
    full_head_mha = _SingleFileDownloadableDatasetLoader('FullHead.mhd')
    return full_head_mha, full_head_raw


_dataset_full_head = _MultiFileDownloadableDatasetLoader(_full_head_files_func)


@_deprecate_positional_args
def download_nek5000(load=True):  # noqa: FBT002
    """Download 2D nek5000 data example.

    .. versionadded:: 0.45.0

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned. True requires
        vtk >= 9.3.

    Returns
    -------
    pyvista.UnstructuredGrid | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_nek5000()
    >>> dataset.plot(scalars='Velocity', cpos='xy')

    .. seealso::

        :ref:`Nek5000 Dataset <nek5000_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    # Silence info messages about 2D mesh found
    with pyvista.vtk_verbosity('off'):
        return _download_dataset(_dataset_nek5000, load=load)


def _nek_5000_download():
    nek5000 = _SingleFileDownloadableDatasetLoader('nek5000/eddy_uv.nek5000')
    data_files = [_DownloadableFile(f'nek5000/eddy_uv0.f{str(i).zfill(5)}') for i in range(1, 12)]
    return (nek5000, *data_files)


_dataset_nek5000 = _MultiFileDownloadableDatasetLoader(_nek_5000_download)


@_deprecate_positional_args
def download_biplane(load=True):  # noqa: FBT002
    """Download biplane dataset.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.MultiBlock | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_biplane()
    >>> dataset.plot(cpos='zy', zoom=1.5)

    .. seealso::

        :ref:`Biplane Dataset <biplane_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_biplane, load=load)


_dataset_biplane = _SingleFileDownloadableDatasetLoader('biplane_rms_pressure_bs.exo')


def download_yinyang(*, load=True):
    """Download yinyang dataset.

    .. versionadded:: 0.46.0

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ImageData | str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Load the image and plot it as grayscale pixel cells.

    >>> from pyvista import examples
    >>> dataset = examples.download_yinyang()
    >>> pixel_cells = dataset.points_to_cells()
    >>> pixel_cells.plot(
    ...     cmap='gray',
    ...     clim=[0, 255],
    ...     cpos='xy',
    ...     zoom='tight',
    ...     lighting=False,
    ...     show_scalar_bar=False,
    ...     show_axes=False,
    ... )

    .. seealso::

        :ref:`Yinyang Dataset <yinyang_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_yinyang, load=load)


_dataset_yinyang = _SingleFileDownloadableDatasetLoader('yinyang/Yinyang.png')
