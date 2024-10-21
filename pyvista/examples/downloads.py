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
import logging
import os
from pathlib import Path
from pathlib import PureWindowsPath
import shutil
import warnings

import numpy as np
import pooch
from pooch import Unzip
from pooch.utils import get_logger

import pyvista
from pyvista.core import _vtk_core as _vtk
from pyvista.core.errors import PyVistaDeprecationWarning
from pyvista.core.errors import VTKVersionError
from pyvista.core.utilities.fileio import get_ext
from pyvista.core.utilities.fileio import read
from pyvista.core.utilities.fileio import read_texture
from pyvista.examples._dataset_loader import _download_dataset
from pyvista.examples._dataset_loader import _DownloadableFile
from pyvista.examples._dataset_loader import _load_and_merge
from pyvista.examples._dataset_loader import _load_as_cubemap
from pyvista.examples._dataset_loader import _load_as_multiblock
from pyvista.examples._dataset_loader import _MultiFileDownloadableDatasetLoader
from pyvista.examples._dataset_loader import _SingleFileDownloadableDatasetLoader

# disable pooch verbose logging
POOCH_LOGGER = get_logger()
POOCH_LOGGER.setLevel(logging.CRITICAL)


CACHE_VERSION = 3


# If available, a local vtk-data instance will be used for examples
if 'PYVISTA_VTK_DATA' in os.environ:  # pragma: no cover
    _path = os.environ['PYVISTA_VTK_DATA']

    if Path(_path).name != 'Data':
        # append 'Data' if user does not provide it
        _path = str(Path(_path) / 'Data')

    # pooch assumes this is a URL so we have to take care of this
    if not _path.endswith('/'):
        _path = _path + '/'
    SOURCE = _path
    _FILE_CACHE = True

else:
    SOURCE = 'https://github.com/pyvista/vtk-data/raw/master/Data/'
    _FILE_CACHE = False

# allow user to override the local path
default_user_data_path = str(pooch.os_cache(f'pyvista_{CACHE_VERSION}'))
if 'PYVISTA_USERDATA_PATH' in os.environ:  # pragma: no cover
    if not Path(os.environ['PYVISTA_USERDATA_PATH']).is_dir():
        warnings.warn('Ignoring invalid {PYVISTA_USERDATA_PATH')
        USER_DATA_PATH = default_user_data_path
    else:
        USER_DATA_PATH = os.environ['PYVISTA_USERDATA_PATH']
else:
    # use default pooch path
    USER_DATA_PATH = default_user_data_path

    # provide helpful message if pooch path is inaccessible
    if not Path(USER_DATA_PATH).is_dir():  # pragma: no cover
        try:
            Path(USER_DATA_PATH, exist_ok=True).mkdir()
            if not os.access(USER_DATA_PATH, os.W_OK):
                raise OSError
        except (PermissionError, OSError):
            # Warn, don't raise just in case there's an environment issue.
            warnings.warn(
                f'Unable to access {USER_DATA_PATH}. Manually specify the PyVista'
                'examples cache with the PYVISTA_USERDATA_PATH environment variable.',
            )

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
        if os.name == 'nt':  # pragma: no cover
            fname = PureWindowsPath(fname).as_posix()
        # ignore mac hidden directories
        if '/__MACOSX/' in fname:
            continue
        if fname.endswith(target_path):
            found_fnames.append(fname)

    if len(found_fnames) == 1:
        return found_fnames[0]

    if len(found_fnames) > 1:
        files_str = '\n'.join(found_fnames)
        raise RuntimeError(f'Ambiguous "{target_path}". Multiple matches found:\n{files_str}')

    files_str = '\n'.join(fnames)
    raise FileNotFoundError(f'Missing "{target_path}" from archive. Archive contains:\n{files_str}')


def _file_copier(input_file, output_file, *args, **kwargs):
    """Copy a file from a local directory to the output path."""
    if not Path(input_file).is_file():
        raise FileNotFoundError(f"'{input_file}' not found within PYVISTA_VTK_DATA '{SOURCE}'")
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


def _download_archive(filename, target_file=None):  # pragma: no cover
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
    return folder if Path(folder).is_dir() else _download_archive(filename, target_file=target_file)


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


def _download_and_read(filename, texture=False, file_format=None, load=True):
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
        raise ValueError('Cannot download and read an archive file')

    saved_file = download_file(filename)
    if not load:
        return saved_file
    if texture:
        return read_texture(saved_file)
    return read(saved_file, file_format=file_format)


def download_masonry_texture(load=True):  # pragma: no cover
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
    read_func=read_texture,
)


def download_usa_texture(load=True):  # pragma: no cover
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
    >>> dataset.plot(cpos="xy")

    .. seealso::

        :ref:`Usa Texture Dataset <usa_texture_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Usa Dataset <usa_dataset>`

    """
    return _download_dataset(_dataset_usa_texture, load=load)


_dataset_usa_texture = _SingleFileDownloadableDatasetLoader('usa_image.jpg', read_func=read_texture)


def download_puppy_texture(load=True):  # pragma: no cover
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
    >>> dataset.plot(cpos="xy")

    .. seealso::

        :ref:`Puppy Texture Dataset <puppy_texture_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Puppy Dataset <puppy_dataset>`

        :ref:`texture_example`
            Example which uses this dataset.

    """
    return _download_dataset(_dataset_puppy_texture, load=load)


_dataset_puppy_texture = _SingleFileDownloadableDatasetLoader('puppy.jpg', read_func=read_texture)


def download_puppy(load=True):  # pragma: no cover
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

    """
    return _download_dataset(_dataset_puppy, load=load)


_dataset_puppy = _SingleFileDownloadableDatasetLoader('puppy.jpg')


def download_usa(load=True):  # pragma: no cover
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
    >>> dataset.plot(style="wireframe", cpos="xy")

    .. seealso::

        :ref:`Usa Dataset <usa_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Usa Texture Dataset <usa_texture_dataset>`

    """
    return _download_dataset(_dataset_usa, load=load)


_dataset_usa = _SingleFileDownloadableDatasetLoader('usa.vtk')


def download_st_helens(load=True):  # pragma: no cover
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
    >>> dataset.plot(cmap="gist_earth")

    .. seealso::

        :ref:`St Helens Dataset <st_helens_dataset>`
            See this dataset in the Dataset Gallery for more info.

        This dataset is used in the following examples:

        * :ref:`colormap_example`
        * :ref:`lighting_properties_example`
        * :ref:`plot_opacity_example`
        * :ref:`orbiting_example`
        * :ref:`plot_over_line_example`
        * :ref:`plotter_lighting_example`
        * :ref:`themes_example`

    """
    return _download_dataset(_dataset_st_helens, load=load)


_dataset_st_helens = _SingleFileDownloadableDatasetLoader('SainteHelens.dem')


def download_bunny(load=True):  # pragma: no cover
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
    >>> dataset.plot(cpos="xy")

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


def download_bunny_coarse(load=True):  # pragma: no cover
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
    >>> dataset.plot(cpos="xy")

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


def download_cow(load=True):  # pragma: no cover
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
    >>> dataset.plot(cpos="xy")

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


def download_cow_head(load=True):  # pragma: no cover
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
    >>> dataset.plot(cpos="xy")

    .. seealso::

        :ref:`Cow Head Dataset <cow_head_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Cow Dataset <cow_dataset>`

    """
    return _download_dataset(_dataset_cow_head, load=load)


_dataset_cow_head = _SingleFileDownloadableDatasetLoader('cowHead.vtp')


def download_faults(load=True):  # pragma: no cover
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


def download_tensors(load=True):  # pragma: no cover
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


def download_head(load=True):  # pragma: no cover
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
    >>> _ = pl.add_volume(dataset, cmap="cool", opacity="sigmoid_6")
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


def download_head_2(load=True):  # pragma: no cover
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
    >>> _ = pl.add_volume(dataset, cmap="cool", opacity="sigmoid_6")
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


def download_bolt_nut(load=True):  # pragma: no cover
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
    ...     cmap="coolwarm",
    ...     opacity="sigmoid_5",
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


def _bolt_nut_files_func():  # pragma: no cover
    # Multiple mesh files are loaded for this example
    bolt = _SingleFileDownloadableDatasetLoader('bolt.slc')
    nut = _SingleFileDownloadableDatasetLoader('nut.slc')
    return bolt, nut


_dataset_bolt_nut = _MultiFileDownloadableDatasetLoader(
    _bolt_nut_files_func,
    load_func=_load_as_multiblock,
)


def download_clown(load=True):  # pragma: no cover
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


def download_topo_global(load=True):  # pragma: no cover
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
    >>> dataset.plot(cmap="gist_earth")

    .. seealso::

        :ref:`Topo Global Dataset <topo_global_dataset>`
            See this dataset in the Dataset Gallery for more info.

        This dataset is used in the following examples:

        * :ref:`surface_normal_example`
        * :ref:`background_image_example`

    """
    return _download_dataset(_dataset_topo_global, load=load)


_dataset_topo_global = _SingleFileDownloadableDatasetLoader('EarthModels/ETOPO_10min_Ice.vtp')


def download_topo_land(load=True):  # pragma: no cover
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
    >>> dataset.plot(
    ...     clim=[-2000, 3000], cmap="gist_earth", show_scalar_bar=False
    ... )

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


def download_coastlines(load=True):  # pragma: no cover
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


def download_knee(load=True):  # pragma: no cover
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
    >>> dataset.plot(cpos="xy", show_scalar_bar=False)

    .. seealso::

        :ref:`Knee Dataset <knee_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Knee Full Dataset <knee_full_dataset>`

        :ref:`medical_dataset_gallery`
            Browse other medical datasets.

        This dataset is used in the following examples:

        * :ref:`plot_opacity_example`
        * :ref:`volume_rendering_example`
        * :ref:`slider_bar_widget_example`

    """
    return _download_dataset(_dataset_knee, load=load)


_dataset_knee = _SingleFileDownloadableDatasetLoader('DICOM_KNEE.dcm')


def download_knee_full(load=True):  # pragma: no cover
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
    >>> dataset.plot(
    ...     volume=True, cmap="bone", cpos=cpos, show_scalar_bar=False
    ... )

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


def download_lidar(load=True):  # pragma: no cover
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
    >>> dataset.plot(cmap="gist_earth")

    .. seealso::

        :ref:`Lidar Dataset <lidar_dataset>`
            See this dataset in the Dataset Gallery for more info.

        This dataset is used in the following examples:

        * :ref:`create_point_cloud`
        * :ref:`edl`

    """
    return _download_dataset(_dataset_lidar, load=load)


_dataset_lidar = _SingleFileDownloadableDatasetLoader('kafadar-lidar-interp.vtp')


def download_exodus(load=True):  # pragma: no cover
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


def download_nefertiti(load=True):  # pragma: no cover
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
    >>> dataset.plot(cpos="xz")

    .. seealso::

        :ref:`Nefertiti Dataset <nefertiti_dataset>`
            See this dataset in the Dataset Gallery for more info.

        This dataset is used in the following examples:

        * :ref:`surface_normal_example`
        * :ref:`extract_edges_example`
        * :ref:`show_edges_example`
        * :ref:`edl`
        * :ref:`pbr_example`
        * :ref:`box_widget_example`

    """
    return _download_dataset(_dataset_nefertiti, load=load)


_dataset_nefertiti = _SingleFileDownloadableDatasetLoader(
    'nefertiti.ply.zip',
    target_file='nefertiti.ply',
)


def download_blood_vessels(load=True):  # pragma: no cover
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
        * :ref:`integrate_example`

    """
    return _download_dataset(_dataset_blood_vessels, load=load)


def _blood_vessels_load_func(obj):  # pragma: no cover
    obj.set_active_vectors('velocity')
    return obj


_dataset_blood_vessels = _SingleFileDownloadableDatasetLoader(
    'pvtu_blood_vessels/blood_vessels.zip',
    target_file='T0000000500.pvtu',
    load_func=_blood_vessels_load_func,
)


def download_iron_protein(load=True):  # pragma: no cover
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


def download_tetrahedron(load=True):  # pragma: no cover
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


def download_saddle_surface(load=True):  # pragma: no cover
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


def download_sparse_points(load=True):  # pragma: no cover
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
    >>> dataset.plot(
    ...     scalars="val", render_points_as_spheres=True, point_size=50
    ... )

    .. seealso::

        :ref:`Sparse Points Dataset <sparse_points_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`interpolate_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_sparse_points, load=load)


def _sparse_points_reader(saved_file):  # pragma: no cover
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


def download_foot_bones(load=True):  # pragma: no cover
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

        :ref:`voxelize_surface_mesh_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_foot_bones, load=load)


_dataset_foot_bones = _SingleFileDownloadableDatasetLoader('fsu/footbones.ply')


def download_guitar(load=True):  # pragma: no cover
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


def download_quadratic_pyramid(load=True):  # pragma: no cover
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


def download_bird(load=True):  # pragma: no cover
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
    >>> dataset.plot(rgba=True, cpos="xy")

    .. seealso::

        :ref:`Bird Dataset <bird_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Bird Texture Dataset <bird_texture_dataset>`

    """
    return _download_dataset(_dataset_bird, load=load)


_dataset_bird = _SingleFileDownloadableDatasetLoader('Pileated.jpg')


def download_bird_texture(load=True):  # pragma: no cover
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
    >>> dataset.plot(cpos="xy")

    .. seealso::

        :ref:`Bird Texture Dataset <bird_texture_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Bird Dataset <bird_dataset>`

    """
    return _download_dataset(_dataset_bird_texture, load=load)


_dataset_bird_texture = _SingleFileDownloadableDatasetLoader('Pileated.jpg', read_func=read_texture)


def download_office(load=True):  # pragma: no cover
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


def download_horse_points(load=True):  # pragma: no cover
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


def download_horse(load=True):  # pragma: no cover
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

        :ref:`disabling_mesh_lighting_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_horse, load=load)


_dataset_horse = _SingleFileDownloadableDatasetLoader('horse.vtp')


def download_cake_easy(load=True):  # pragma: no cover
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
    >>> dataset.plot(rgba=True, cpos="xy")

    .. seealso::

        :ref:`Cake Easy Dataset <cake_easy_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Cake Easy Texture Dataset <cake_easy_texture_dataset>`

    """
    return _download_dataset(_dataset_cake_easy, load=load)


_dataset_cake_easy = _SingleFileDownloadableDatasetLoader('cake_easy.jpg')


def download_cake_easy_texture(load=True):  # pragma: no cover
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
    >>> dataset.plot(cpos="xy")

    .. seealso::

        :ref:`Cake Easy Texture Dataset <cake_easy_texture_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Cake Easy Dataset <cake_easy_dataset>`

    """
    return _download_dataset(_dataset_cake_easy_texture, load=load)


_dataset_cake_easy_texture = _SingleFileDownloadableDatasetLoader(
    'cake_easy.jpg',
    read_func=read_texture,
)


def download_rectilinear_grid(load=True):  # pragma: no cover
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


def download_gourds(zoom=False, load=True):  # pragma: no cover
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
    >>> dataset.plot(rgba=True, cpos="xy")

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


def download_gourds_texture(zoom=False, load=True):  # pragma: no cover
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
    >>> dataset.plot(cpos="xy")

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
_dataset_gourds_texture = _SingleFileDownloadableDatasetLoader('Gourds.png', read_func=read_texture)
__gourds2_texture = _SingleFileDownloadableDatasetLoader('Gourds2.jpg', read_func=read_texture)


def download_gourds_pnm(load=True):  # pragma: no cover
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
    >>> dataset.plot(rgba=True, cpos="xy")

    .. seealso::

        :ref:`Gourds Pnm Dataset <gourds_pnm_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Gourds Dataset <gourds_dataset>`

        :ref:`Gourds Texture Dataset <gourds_texture_dataset>`

    """
    return _download_dataset(_dataset_gourds_pnm, load=load)


_dataset_gourds_pnm = _SingleFileDownloadableDatasetLoader('Gourds.pnm')


def download_unstructured_grid(load=True):  # pragma: no cover
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


def download_letter_k(load=True):  # pragma: no cover
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
    >>> dataset.plot(cpos="xy")

    .. seealso::

        :ref:`Letter K Dataset <letter_k_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Letter A Dataset <letter_a_dataset>`

    """
    return _download_dataset(_dataset_letter_k, load=load)


_dataset_letter_k = _SingleFileDownloadableDatasetLoader('k.vtk')


def download_letter_a(load=True):  # pragma: no cover
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
    >>> dataset.plot(cpos="xy", show_edges=True)

    .. seealso::

        :ref:`Letter A Dataset <letter_a_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Letter K Dataset <letter_k_dataset>`

        :ref:`cell_centers_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_letter_a, load=load)


_dataset_letter_a = _SingleFileDownloadableDatasetLoader('a_grid.vtk')


def download_poly_line(load=True):  # pragma: no cover
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


def download_cad_model(load=True):  # pragma: no cover
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


def download_frog(load=True):  # pragma: no cover
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

        :ref:`Frog Tissue Dataset <frog_tissue_dataset>`
            Segmentation labels associated with this dataset.

        :ref:`medical_dataset_gallery`
            Browse other medical datasets.

        :ref:`volume_rendering_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_frog, load=load)


def _frog_files_func():  # pragma: no cover
    # Multiple files needed for read, but only one gets loaded
    frog_zraw = _DownloadableFile('froggy/frog.zraw')
    frog_mhd = _SingleFileDownloadableDatasetLoader('froggy/frog.mhd')
    return frog_mhd, frog_zraw


_dataset_frog = _MultiFileDownloadableDatasetLoader(_frog_files_func)


def download_frog_tissue(load=True):  # pragma: no cover
    """Download frog tissue dataset.

    This dataset contains tissue segmentation labels for the frog dataset.

    .. deprecated:: 0.44.0

        This example does not load correctly on some systems and has been deprecated.
        Use :func:`~pyvista.examples.load_frog_tissues` instead.

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.ImageData | str
        DataSet or filename depending on ``load``.

    .. seealso::

        :ref:`Frog Tissue Dataset <frog_tissue_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Frog Dataset <frog_dataset>`

        :ref:`medical_dataset_gallery`
            Browse other medical datasets.

    """
    # Deprecated on v0.44.0, estimated removal on v0.47.0
    warnings.warn(
        'This example is deprecated and will be removed in v0.47.0. Use `load_frog_tissues` instead.',
        PyVistaDeprecationWarning,
    )
    if pyvista._version.version_info >= (0, 47):
        raise RuntimeError('Remove this deprecated function')

    return _download_dataset(_dataset_frog_tissue, load=load)


def _frog_tissue_files_func():
    # Multiple files needed for read, but only one gets loaded
    frog_tissue_zraw = _DownloadableFile('froggy/frogtissue.zraw')
    frog_tissue_mhd = _SingleFileDownloadableDatasetLoader('froggy/frogtissue.mhd')
    return frog_tissue_mhd, frog_tissue_zraw


_dataset_frog_tissue = _MultiFileDownloadableDatasetLoader(_frog_tissue_files_func)


def download_chest(load=True):  # pragma: no cover
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
    >>> dataset.plot(cpos="xy")

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


def download_brain_atlas_with_sides(load=True):  # pragma: no cover
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


def download_prostate(load=True):  # pragma: no cover
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
    >>> dataset.plot(cpos="xy")

    .. seealso::

        :ref:`Prostate Dataset <prostate_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`medical_dataset_gallery`
            Browse other medical datasets.

    """
    return _download_dataset(_dataset_prostate, load=load)


_dataset_prostate = _SingleFileDownloadableDatasetLoader('prostate.img')


def download_filled_contours(load=True):  # pragma: no cover
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
    >>> dataset.plot(cpos="xy")

    .. seealso::

        :ref:`Filled Contours Dataset <filled_contours_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_filled_contours, load=load)


_dataset_filled_contours = _SingleFileDownloadableDatasetLoader('filledContours.vtp')


def download_doorman(load=True):  # pragma: no cover
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
    >>> dataset.plot(cpos="xy")

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


def download_mug(load=True):  # pragma: no cover
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


def download_oblique_cone(load=True):  # pragma: no cover
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


def download_emoji(load=True):  # pragma: no cover
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
    >>> dataset.plot(rgba=True, cpos="xy")

    .. seealso::

        :ref:`Emoji Dataset <emoji_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Emoji Texture Dataset <emoji_texture_dataset>`

    """
    return _download_dataset(_dataset_emoji, load=load)


_dataset_emoji = _SingleFileDownloadableDatasetLoader('emote.jpg')


def download_emoji_texture(load=True):  # pragma: no cover
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
    >>> dataset.plot(cpos="xy")

    .. seealso::

        :ref:`Emoji Texture Dataset <emoji_texture_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Emoji Dataset <emoji_dataset>`

    """
    return _download_dataset(_dataset_emoji_texture, load=load)


_dataset_emoji_texture = _SingleFileDownloadableDatasetLoader('emote.jpg', read_func=read_texture)


def download_teapot(load=True):  # pragma: no cover
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
    >>> dataset.plot(cpos="xy")

    .. seealso::

        :ref:`Teapot Dataset <teapot_dataset>`
            See this dataset in the Dataset Gallery for more info.

        This dataset is used in the following examples:

        * :ref:`read_file_example`
        * :ref:`cell_centers_example`

    """
    return _download_dataset(_dataset_teapot, load=load)


_dataset_teapot = _SingleFileDownloadableDatasetLoader('teapot.g')


def download_brain(load=True):  # pragma: no cover
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


def download_structured_grid(load=True):  # pragma: no cover
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


def download_structured_grid_two(load=True):  # pragma: no cover
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


def download_trumpet(load=True):  # pragma: no cover
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


def download_face(load=True):  # pragma: no cover
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


def download_sky_box_nz(load=True):  # pragma: no cover
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
    >>> dataset.plot(rgba=True, cpos="xy")

    .. seealso::

        :ref:`Sky Box Nz Dataset <sky_box_nz_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Sky Box Nz Texture Dataset <sky_box_nz_texture_dataset>`

        :ref:`Sky Box Cube Map Dataset <sky_box_cube_map_dataset>`

    """
    return _download_dataset(_dataset_sky_box_nz, load=load)


_dataset_sky_box_nz = _SingleFileDownloadableDatasetLoader('skybox-nz.jpg')


def download_sky_box_nz_texture(load=True):  # pragma: no cover
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
    >>> dataset.plot(cpos="xy")

    .. seealso::

        :ref:`Sky Box Nz Texture Dataset <sky_box_nz_texture_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Sky Box Nz Dataset <sky_box_nz_dataset>`

        :ref:`Sky Box Cube Map Dataset <sky_box_cube_map_dataset>`

    """
    return _download_dataset(_dataset_sky_box_nz_texture, load=load)


_dataset_sky_box_nz_texture = _SingleFileDownloadableDatasetLoader(
    'skybox-nz.jpg',
    read_func=read_texture,
)


def download_disc_quads(load=True):  # pragma: no cover
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


def download_honolulu(load=True):  # pragma: no cover
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
    ...     cmap="gist_earth",
    ...     clim=[-50, 800],
    ... )

    .. seealso::

        :ref:`Honolulu Dataset <honolulu_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_honolulu, load=load)


_dataset_honolulu = _SingleFileDownloadableDatasetLoader('honolulu.vtk')


def download_motor(load=True):  # pragma: no cover
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


def download_tri_quadratic_hexahedron(load=True):  # pragma: no cover
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


def _tri_quadratic_hexahedron_load_func(dataset):  # pragma: no cover
    dataset.clear_data()
    return dataset


_dataset_tri_quadratic_hexahedron = _SingleFileDownloadableDatasetLoader(
    'TriQuadraticHexahedron.vtu',
    load_func=_tri_quadratic_hexahedron_load_func,
)


def download_human(load=True):  # pragma: no cover
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


def download_vtk(load=True):  # pragma: no cover
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
    >>> dataset.plot(cpos="xy", line_width=5)

    .. seealso::

        :ref:`Vtk Dataset <vtk_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Vtk Logo Dataset <vtk_logo_dataset>`

    """
    return _download_dataset(_dataset_vtk, load=load)


_dataset_vtk = _SingleFileDownloadableDatasetLoader('vtk.vtp')


def download_spider(load=True):  # pragma: no cover
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


def download_carotid(load=True):  # pragma: no cover
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


def _carotid_load_func(mesh):  # pragma: no cover
    mesh.set_active_scalars('scalars')
    mesh.set_active_vectors('vectors')
    return mesh


_dataset_carotid = _SingleFileDownloadableDatasetLoader('carotid.vtk', load_func=_carotid_load_func)


def download_blow(load=True):  # pragma: no cover
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


def download_shark(load=True):  # pragma: no cover
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


def download_great_white_shark(load=True):  # pragma: no cover
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


def download_grey_nurse_shark(load=True):  # pragma: no cover
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


def download_dragon(load=True):  # pragma: no cover
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
    >>> dataset.plot(cpos="xy")

    .. seealso::

        :ref:`Dragon Dataset <dragon_dataset>`
            See this dataset in the Dataset Gallery for more info.

        This dataset is used in the following examples:

        * :ref:`floors_example`
        * :ref:`orbiting_example`
        * :ref:`silhouette_example`
        * :ref:`light_shadows_example`

    """
    return _download_dataset(_dataset_dragon, load=load)


_dataset_dragon = _SingleFileDownloadableDatasetLoader('dragon.ply')


def download_armadillo(load=True):  # pragma: no cover
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


def download_gears(load=True):  # pragma: no cover
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
    >>> for i, body in enumerate(bodies):  # pragma: no cover
    ...     bid = np.empty(body.n_points)
    ...     bid[:] = i
    ...     body.point_data["Body ID"] = bid
    ...
    >>> bodies.plot(cmap='jet')

    .. seealso::

        :ref:`Gears Dataset <gears_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_gears, load=load)


_dataset_gears = _SingleFileDownloadableDatasetLoader('gears.stl')


def download_torso(load=True):  # pragma: no cover
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
    >>> dataset.plot(cpos="xz")

    .. seealso::

        :ref:`Torso Dataset <torso_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_torso, load=load)


_dataset_torso = _SingleFileDownloadableDatasetLoader('Torso.vtp')


def download_kitchen(split=False, load=True):  # pragma: no cover
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


def _kitchen_split_load_func(mesh):  # pragma: no cover
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
    for key, extent in extents.items():  # pragma: no cover
        alg = _vtk.vtkStructuredGridGeometryFilter()
        alg.SetInputDataObject(mesh)
        alg.SetExtent(extent)
        alg.Update()
        result = pyvista.core.filters._get_output(alg)
        kitchen[key] = result
    return kitchen


_dataset_kitchen = _SingleFileDownloadableDatasetLoader('kitchen.vtk')
__kitchen_split = _SingleFileDownloadableDatasetLoader(
    'kitchen.vtk',
    load_func=_kitchen_split_load_func,
)


def download_tetra_dc_mesh(load=True):  # pragma: no cover
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


def _tetra_dc_mesh_files_func():  # pragma: no cover
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


def download_model_with_variance(load=True):  # pragma: no cover
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

        :ref:`plot_opacity_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_model_with_variance, load=load)


_dataset_model_with_variance = _SingleFileDownloadableDatasetLoader('model_with_variance.vtu')


def download_thermal_probes(load=True):  # pragma: no cover
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
    >>> dataset.plot(
    ...     render_points_as_spheres=True, point_size=5, cpos="xy"
    ... )

    .. seealso::

        :ref:`Thermal Probes Dataset <thermal_probes_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`interpolate_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_thermal_probes, load=load)


_dataset_thermal_probes = _SingleFileDownloadableDatasetLoader('probes.vtp')


def download_carburetor(load=True):  # pragma: no cover
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


def download_turbine_blade(load=True):  # pragma: no cover
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


def download_pine_roots(load=True):  # pragma: no cover
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


def download_crater_topo(load=True):  # pragma: no cover
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
    >>> dataset.plot(cmap="gist_earth", cpos="xy")

    .. seealso::

        :ref:`Crater Topo Dataset <crater_topo_dataset>`
            See this dataset in the Dataset Gallery for more info.

        This dataset is used in the following examples:

        * :ref:`terrain_following_mesh_example`
        * :ref:`topo_map_example`

    """
    return _download_dataset(_dataset_crater_topo, load=load)


_dataset_crater_topo = _SingleFileDownloadableDatasetLoader('Ruapehu_mag_dem_15m_NZTM.vtk')


def download_crater_imagery(load=True):  # pragma: no cover
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
    read_func=read_texture,
)


def download_dolfin(load=True):  # pragma: no cover
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
    >>> dataset.plot(cpos="xy", show_edges=True)

    .. seealso::

        :ref:`Dolfin Dataset <dolfin_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_dolfin, load=load)


_dataset_dolfin = _SingleFileDownloadableDatasetLoader(
    'dolfin_fine.xml',
    read_func=functools.partial(read, file_format='dolfin-xml'),
)


def download_damavand_volcano(load=True):  # pragma: no cover
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
    >>> from pyvista import examples
    >>> cpos = [
    ...     [4.66316700e04, 4.32796241e06, -3.82467050e05],
    ...     [5.52532740e05, 3.98017300e06, -2.47450000e04],
    ...     [4.10000000e-01, -2.90000000e-01, -8.60000000e-01],
    ... ]
    >>> dataset = examples.download_damavand_volcano()
    >>> dataset.plot(
    ...     cpos=cpos, cmap="reds", show_scalar_bar=False, volume=True
    ... )

    .. seealso::

        :ref:`Damavand Volcano Dataset <damavand_volcano_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`volume_rendering_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_damavand_volcano, load=load)


def _damavand_volcano_load_func(volume):  # pragma: no cover
    volume.rename_array('None', 'data')
    return volume


_dataset_damavand_volcano = _SingleFileDownloadableDatasetLoader(
    'damavand-volcano.vtk',
    load_func=_damavand_volcano_load_func,
)


def download_delaunay_example(load=True):  # pragma: no cover
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


def download_embryo(load=True):  # pragma: no cover
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
        * :ref:`orthogonal_slices_example`

    """
    return _download_dataset(_dataset_embryo, load=load)


def _embryo_load_func(dataset):  # pragma: no cover
    # cleanup artifact
    mask = dataset['SLCImage'] == 255
    dataset['SLCImage'][mask] = 0
    return dataset


_dataset_embryo = _SingleFileDownloadableDatasetLoader('embryo.slc', load_func=_embryo_load_func)


def download_antarctica_velocity(load=True):  # pragma: no cover
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
    >>> dataset.plot(
    ...     cpos='xy', clim=[1e-3, 1e4], cmap='Blues', log_scale=True
    ... )

    .. seealso::

        :ref:`Antarctica Velocity Dataset <antarctica_velocity_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`antarctica_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_antarctica_velocity, load=load)


_dataset_antarctica_velocity = _SingleFileDownloadableDatasetLoader('antarctica_velocity.vtp')


def download_room_surface_mesh(load=True):  # pragma: no cover
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


def download_beach(load=True):  # pragma: no cover
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
    >>> dataset.plot(rgba=True, cpos="xy")

    .. seealso::

        :ref:`Beach Dataset <beach_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_beach, load=load)


_dataset_beach = _SingleFileDownloadableDatasetLoader('beach.nrrd')


def download_rgba_texture(load=True):  # pragma: no cover
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
    >>> dataset.plot(cpos="xy")

    .. seealso::

        :ref:`Rgba Texture Dataset <rgba_texture_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`texture_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_rgba_texture, load=load)


_dataset_rgba_texture = _SingleFileDownloadableDatasetLoader(
    'alphachannel.png',
    read_func=read_texture,
)


def download_vtk_logo(load=True):  # pragma: no cover
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
    >>> dataset.plot(cpos="xy")

    .. seealso::

        :ref:`Vtk Logo Dataset <vtk_logo_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Vtk Dataset <vtk_dataset>`

    """
    return _download_dataset(_dataset_vtk_logo, load=load)


_dataset_vtk_logo = _SingleFileDownloadableDatasetLoader('vtk.png', read_func=read_texture)


def download_sky_box_cube_map(load=True):  # pragma: no cover
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


def download_cubemap_park(load=True):  # pragma: no cover
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
    >>> pl.set_environment_texture(dataset, True)
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
    read_func=_load_as_cubemap,
)


def download_cubemap_space_4k(load=True):  # pragma: no cover
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
    >>> pl.set_environment_texture(cubemap, True)
    >>> pl.camera.zoom(0.4)
    >>> _ = pl.add_mesh(
    ...     pv.Sphere(), pbr=True, roughness=0.24, metallic=1.0
    ... )
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
    read_func=_load_as_cubemap,
)


def download_cubemap_space_16k(load=True):  # pragma: no cover
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
    >>> pl.set_environment_texture(cubemap, True)
    >>> pl.camera.zoom(0.4)
    >>> _ = pl.add_mesh(
    ...     pv.Sphere(), pbr=True, roughness=0.24, metallic=1.0
    ... )
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
    read_func=_load_as_cubemap,
)


def download_backward_facing_step(load=True):  # pragma: no cover
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


def download_gpr_data_array(load=True):  # pragma: no cover
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

        :ref:`create_draped_surf_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_gpr_data_array, load=load)


_dataset_gpr_data_array = _SingleFileDownloadableDatasetLoader(
    'gpr-example/data.npy',
    read_func=np.load,
)


def download_gpr_path(load=True):  # pragma: no cover
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

        :ref:`create_draped_surf_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_gpr_path, load=load)


_dataset_gpr_path = _SingleFileDownloadableDatasetLoader(
    'gpr-example/path.txt',
    read_func=functools.partial(np.loadtxt, skiprows=1),  # type: ignore[arg-type]
    load_func=pyvista.PolyData,
)


def download_woman(load=True):  # pragma: no cover
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


def download_lobster(load=True):  # pragma: no cover
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


def download_face2(load=True):  # pragma: no cover
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


def download_urn(load=True):  # pragma: no cover
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


def download_pepper(load=True):  # pragma: no cover
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


def download_drill(load=True):  # pragma: no cover
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
    return _download_dataset(_dataset_drill, load=load)


_dataset_drill = _SingleFileDownloadableDatasetLoader('drill.obj')


def download_action_figure(load=True):  # pragma: no cover
    """Download scan of an action figure.

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
    Show the action figure example. This also demonstrates how to use
    physically based rendering and lighting to make a good looking
    plot.

    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> dataset = examples.download_action_figure()
    >>> _ = dataset.clean(inplace=True)
    >>> pl = pv.Plotter(lighting=None)
    >>> pl.add_light(pv.Light((30, 10, 10)))
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
    return _download_dataset(_dataset_action_figure, load=load)


_dataset_action_figure = _SingleFileDownloadableDatasetLoader('tigerfighter.obj')


def download_notch_stress(load=True):  # pragma: no cover
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


def download_notch_displacement(load=True):  # pragma: no cover
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


def download_louis_louvre(load=True):  # pragma: no cover
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
    >>> pl.add_light(pv.Light((10, -10, 10)))
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


def download_cylinder_crossflow(load=True):  # pragma: no cover
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

        :ref:`2d_streamlines_example`
            Example using this dataset.

    """
    return _download_dataset(_dataset_cylinder_crossflow, load=load)


def _cylinder_crossflow_files_func():  # pragma: no cover
    case = _SingleFileDownloadableDatasetLoader('EnSight/CylinderCrossflow/cylinder_Re35.case')
    geo = _DownloadableFile('EnSight/CylinderCrossflow/cylinder_Re35.geo')
    scl1 = _DownloadableFile('EnSight/CylinderCrossflow/cylinder_Re35.scl1')
    scl2 = _DownloadableFile('EnSight/CylinderCrossflow/cylinder_Re35.scl2')
    vel = _DownloadableFile('EnSight/CylinderCrossflow/cylinder_Re35.vel')
    return case, geo, scl1, scl2, vel


_dataset_cylinder_crossflow = _MultiFileDownloadableDatasetLoader(
    files_func=_cylinder_crossflow_files_func,
)


def download_naca(load=True):  # pragma: no cover
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
    >>> dataset.plot(cpos=cpos, cmap="jet")

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


def download_lshape(load=True):  # pragma: no cover
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
    >>> mesh = examples.download_lshape()["all"]
    >>> warped = mesh.warp_by_vector(factor=30)
    >>> warped.plot(scalars="displacement")

    .. seealso::

        :ref:`Lshape Dataset <lshape_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_lshape, load=load)


def _lshape_files_func():  # pragma: no cover
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


def download_wavy(load=True):  # pragma: no cover
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


def download_single_sphere_animation(load=True):  # pragma: no cover
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
    ...     _ = plotter.add_text(f"Time: {time_value:.0f}", color="black")
    ...     plotter.write_frame()
    ...     plotter.clear()
    ...     plotter.enable_lightkit()
    ...
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


def download_dual_sphere_animation(load=True):  # pragma: no cover
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
    ...     _ = plotter.add_text(f"Time: {time_value:.0f}", color="black")
    ...     plotter.write_frame()
    ...     plotter.clear()
    ...     plotter.enable_lightkit()
    ...
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


def download_osmnx_graph(load=True):  # pragma: no cover
    """Load a simple street map from Open Street Map.

    Generated from:

    .. code:: python

        >>> import osmnx as ox  # doctest:+SKIP
        >>> address = 'Holzgerlingen DE'  # doctest:+SKIP
        >>> graph = ox.graph_from_address(
        ...     address, dist=500, network_type='drive'
        ... )  # doctest:+SKIP
        >>> pickle.dump(
        ...     graph, open('osmnx_graph.p', 'wb')
        ... )  # doctest:+SKIP

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    networkx.classes.multidigraph.MultiDiGraph
        An osmnx graph of the streets of Holzgerlingen, Germany.

    Examples
    --------
    >>> from pyvista import examples
    >>> graph = examples.download_osmnx_graph()  # doctest:+SKIP

    """
    # Deprecated on v0.44.0, estimated removal on v0.47.0
    warnings.warn(
        '`download_osmnx_graph` is deprecated and will be removed in v0.47.0. Please use https://github.com/pyvista/pyvista-osmnx.',
        PyVistaDeprecationWarning,
    )
    if pyvista._version.version_info >= (0, 47):
        raise RuntimeError('Remove this deprecated function')
    try:
        import osmnx  # noqa: F401
    except ImportError:
        raise ImportError('Install `osmnx` to use this example')
    return _download_dataset(_dataset_osmnx_graph, load=load)


def _osmnx_graph_read_func(filename):  # pragma: no cover
    import pickle

    return pickle.load(Path(filename).open('rb'))  # noqa: SIM115


_dataset_osmnx_graph = _SingleFileDownloadableDatasetLoader(
    'osmnx_graph.p',
    read_func=_osmnx_graph_read_func,
)


def download_cavity(load=True):  # pragma: no cover
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

        :ref:`Osmnx Graph Dataset <osmnx_graph_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`openfoam_example`
            Full example using this dataset.

    """
    return _download_dataset(_dataset_cavity, load=load)


_dataset_cavity = _SingleFileDownloadableDatasetLoader(
    'OpenFOAM.zip',
    target_file='cavity/case.foam',
)


def download_openfoam_tubes(load=True):  # pragma: no cover
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


def _openfoam_tubes_read_func(filename):  # pragma: no cover
    reader = pyvista.OpenFOAMReader(filename)
    reader.set_active_time_value(1000)
    return reader.read()


_dataset_openfoam_tubes = _SingleFileDownloadableDatasetLoader(
    'fvm/turbo_incompressible/Turbo-Incompressible_3-Run_1-SOLUTION_FIELDS.zip',
    target_file='case.foam',
    read_func=_openfoam_tubes_read_func,
)


def download_lucy(load=True):  # pragma: no cover
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


def download_pump_bracket(load=True):  # pragma: no cover
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


def download_electronics_cooling(load=True):  # pragma: no cover
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


def _electronics_cooling_files_func():  # pragma: no cover
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


def download_can(partial=False, load=True):  # pragma: no cover
    """Download the can dataset mesh.

    File obtained from `Kitware <https://www.kitware.com/>`_. Used
    for testing hdf files.

    Parameters
    ----------
    partial : bool, default: False
        Load part of the dataset.

    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData, str, or list[str]
        The example ParaView can DataSet or file path(s).

    Examples
    --------
    Plot the can dataset.

    >>> from pyvista import examples
    >>> import pyvista as pv
    >>> dataset = examples.download_can()  # doctest:+SKIP
    >>> dataset.plot(scalars='VEL', smooth_shading=True)  # doctest:+SKIP

    .. seealso::

        :ref:`Can Dataset <can_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Can Crushed Hdf Dataset <can_crushed_hdf_dataset>`

        :ref:`Can Crushed Vtu Dataset <can_crushed_vtu_dataset>`

    """
    if partial:
        return _download_dataset(__can_partial, load=load)
    else:
        return _download_dataset(_dataset_can, load=load)


def _dataset_can_files_func():  # pragma: no cover
    if pyvista.vtk_version_info > (9, 1):
        raise VTKVersionError(
            'This example file is deprecated for VTK v9.2.0 and newer. '
            'Use `download_can_crushed_hdf` instead.',
        )
    can_0 = _SingleFileDownloadableDatasetLoader('hdf/can_0.hdf')
    can_1 = _SingleFileDownloadableDatasetLoader('hdf/can_1.hdf')
    can_2 = _SingleFileDownloadableDatasetLoader('hdf/can_2.hdf')
    return can_0, can_1, can_2


_dataset_can = _MultiFileDownloadableDatasetLoader(
    files_func=_dataset_can_files_func,
    load_func=_load_and_merge,
)
__can_partial = _SingleFileDownloadableDatasetLoader('hdf/can_0.hdf')


def download_can_crushed_hdf(load=True):  # pragma: no cover
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

        :ref:`Can Dataset <can_dataset>`

    """
    return _download_dataset(_dataset_can_crushed_hdf, load=load)


_dataset_can_crushed_hdf = _SingleFileDownloadableDatasetLoader('hdf/can-vtu.hdf')


def download_can_crushed_vtu(load=True):  # pragma: no cover
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

        :ref:`Can Dataset <can_dataset>`

    """
    return _download_dataset(_dataset_can_crushed_vtu, load=load)


_dataset_can_crushed_vtu = _SingleFileDownloadableDatasetLoader('can.vtu')


def download_cgns_structured(load=True):  # pragma: no cover
    """Download the structured CGNS dataset mesh.

    Originally downloaded from `CFD General Notation System Example Files
    <https://cgns.github.io/CGNSFiles.html>`_

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


def download_tecplot_ascii(load=True):  # pragma: no cover
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


def download_cgns_multi(load=True):  # pragma: no cover
    """Download a multielement airfoil with a cell centered solution.

    Originally downloaded from `CFD General Notation System Example Files
    <https://cgns.github.io/CGNSFiles.html>`_

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


def _cgns_multi_read_func(filename):  # pragma: no cover
    reader = pyvista.get_reader(filename)
    # disable reading the boundary patch. As of VTK 9.1.0 this generates
    # messages like "Skipping BC_t node: BC_t type 'BCFarfield' not supported
    # yet."
    reader.load_boundary_patch = False
    return reader.read()


_dataset_cgns_multi = _SingleFileDownloadableDatasetLoader(
    'cgns/multi.cgns',
    read_func=_cgns_multi_read_func,
)


def download_dicom_stack(load: bool = True) -> pyvista.ImageData | str:  # pragma: no cover
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
        Maffitt D, Pringle M, Tarbox L, Prior F. The Cancer Imaging Archive (TCIA):  # pragma: no cover
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


def download_parched_canal_4k(load=True):  # pragma: no cover
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
    >>> dataset = examples.download_parched_canal_4k()
    >>> dataset.plot(cpos="xy")

    .. seealso::

        :ref:`Parched Canal 4k Dataset <parched_canal_4k_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_parched_canal_4k, load=load)


_dataset_parched_canal_4k = _SingleFileDownloadableDatasetLoader(
    'parched_canal_4k.hdr',
    read_func=read_texture,
)


def download_cells_nd(load=True):  # pragma: no cover
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
    >>> dataset.plot(cpos="xy")

    .. seealso::

        :ref:`CellsNd Dataset <cells_nd_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_cells_nd, load=load)


_dataset_cells_nd = _SingleFileDownloadableDatasetLoader('cellsnd.ascii.inp')


def download_moonlanding_image(load=True):  # pragma: no cover
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


def download_angular_sector(load=True):  # pragma: no cover
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


def download_mount_damavand(load=True):  # pragma: no cover
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


def download_particles_lethe(load=True):  # pragma: no cover
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


def download_gif_simple(load=True):  # pragma: no cover
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


def download_cloud_dark_matter(load=True):  # pragma: no cover
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

        :ref:`plotting_point_clouds`
            Full example using this dataset

    """
    return _download_dataset(_dataset_cloud_dark_matter, load=load)


_dataset_cloud_dark_matter = _SingleFileDownloadableDatasetLoader(
    'point-clouds/findus23/halo_low_res.npy',
    read_func=np.load,
    load_func=pyvista.PointSet,
)


def download_cloud_dark_matter_dense(load=True):  # pragma: no cover
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

        :ref:`plotting_point_clouds`
            More details on how to plot point clouds.

    """
    return _download_dataset(_dataset_cloud_dark_matter_dense, load=load)


_dataset_cloud_dark_matter_dense = _SingleFileDownloadableDatasetLoader(
    'point-clouds/findus23/halo_high_res.npy',
    read_func=np.load,
    load_func=pyvista.PointSet,
)


def download_stars_cloud_hyg(load=True):  # pragma: no cover
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

        :ref:`plotting_point_clouds`
            More details on how to plot point clouds.

    """
    return _download_dataset(_dataset_stars_cloud_hyg, load=load)


_dataset_stars_cloud_hyg = _SingleFileDownloadableDatasetLoader(
    'point-clouds/hyg-database/stars.vtp',
)


def download_fea_bracket(load=True):  # pragma: no cover
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


def download_fea_hertzian_contact_cylinder(load=True):  # pragma: no cover
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
    >>> grid.plot(
    ...     scalars='PartID', cmap=['green', 'blue'], show_scalar_bar=False
    ... )

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


def download_black_vase(load=True):  # pragma: no cover
    """Download a black vase scan created by Ivan Nikolov.

    The dataset was downloaded from `GGG-BenchmarkSfM: Dataset for Benchmarking
    Close-range SfM Software Performance under Varying Capturing Conditions
    <https://data.mendeley.com/datasets/bzxk2n78s9/4>`_

    Original datasets are under the CC BY 4.0 license.

    For more details, see `Ivan Nikolov Datasets
    <https://github.com/pyvista/vtk-data/tree/master/Data/ivan-nikolov>`_

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
    >>> mesh = examples.download_black_vase()
    >>> mesh.plot()

    Return the statistics of the dataset.

    >>> mesh
    PolyData (...)
      N Cells:    3136652
      N Points:   1611789
      N Strips:   0
      X Bounds:   -1.092e+02, 1.533e+02
      Y Bounds:   -1.200e+02, 1.415e+02
      Z Bounds:   1.666e+01, 4.077e+02
      N Arrays:   0

    .. seealso::

        :ref:`Black Vase Dataset <black_vase_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_black_vase, load=load)


_dataset_black_vase = _SingleFileDownloadableDatasetLoader(
    'ivan-nikolov/blackVase.zip',
    target_file='blackVase.vtp',
)


def download_ivan_angel(load=True):  # pragma: no cover
    """Download a scan of an angel statue created by Ivan Nikolov.

    The dataset was downloaded from `GGG-BenchmarkSfM: Dataset for Benchmarking
    Close-range SfM Software Performance under Varying Capturing Conditions
    <https://data.mendeley.com/datasets/bzxk2n78s9/4>`_

    Original datasets are under the CC BY 4.0 license.

    For more details, see `Ivan Nikolov Datasets
    <https://github.com/pyvista/vtk-data/tree/master/Data/ivan-nikolov>`_

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
      N Cells:    3580454
      N Points:   1811531
      N Strips:   0
      X Bounds:   -1.147e+02, 8.468e+01
      Y Bounds:   -6.996e+01, 9.247e+01
      Z Bounds:   -1.171e+02, 2.052e+02
      N Arrays:   0

    .. seealso::

        :ref:`Ivan Angel Dataset <ivan_angel_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_ivan_angel, load=load)


_dataset_ivan_angel = _SingleFileDownloadableDatasetLoader(
    'ivan-nikolov/Angel.zip',
    target_file='Angel.vtp',
)


def download_bird_bath(load=True):  # pragma: no cover
    """Download a scan of a bird bath created by Ivan Nikolov.

    The dataset was downloaded from `GGG-BenchmarkSfM: Dataset for Benchmarking
    Close-range SfM Software Performance under Varying Capturing Conditions
    <https://data.mendeley.com/datasets/bzxk2n78s9/4>`_

    Original datasets are under the CC BY 4.0 license.

    For more details, see `Ivan Nikolov Datasets
    <https://github.com/pyvista/vtk-data/tree/master/Data/ivan-nikolov>`_

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
    >>> mesh = examples.download_bird_bath()
    >>> mesh.plot()

    Return the statistics of the dataset.

    >>> mesh
    PolyData (...)
      N Cells:    3507935
      N Points:   1831383
      N Strips:   0
      X Bounds:   -1.601e+02, 1.483e+02
      Y Bounds:   -1.521e+02, 1.547e+02
      Z Bounds:   -4.241e+00, 1.409e+02
      N Arrays:   0

    .. seealso::

        :ref:`Bird Bath Dataset <bird_bath_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_bird_bath, load=load)


_dataset_bird_bath = _SingleFileDownloadableDatasetLoader(
    'ivan-nikolov/birdBath.zip',
    target_file='birdBath.vtp',
)


def download_owl(load=True):  # pragma: no cover
    """Download a scan of an owl statue created by Ivan Nikolov.

    The dataset was downloaded from `GGG-BenchmarkSfM: Dataset for Benchmarking
    Close-range SfM Software Performance under Varying Capturing Conditions
    <https://data.mendeley.com/datasets/bzxk2n78s9/4>`_

    Original datasets are under the CC BY 4.0 license.

    For more details, see `Ivan Nikolov Datasets
    <https://github.com/pyvista/vtk-data/tree/master/Data/ivan-nikolov>`_

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
      N Cells:    2440707
      N Points:   1221756
      N Strips:   0
      X Bounds:   -5.834e+01, 7.047e+01
      Y Bounds:   -7.006e+01, 6.658e+01
      Z Bounds:   1.676e+00, 2.013e+02
      N Arrays:   0

    .. seealso::

        :ref:`Owl Dataset <owl_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_owl, load=load)


_dataset_owl = _SingleFileDownloadableDatasetLoader('ivan-nikolov/owl.zip', target_file='owl.vtp')


def download_plastic_vase(load=True):  # pragma: no cover
    """Download a scan of a plastic vase created by Ivan Nikolov.

    The dataset was downloaded from `GGG-BenchmarkSfM: Dataset for Benchmarking
    Close-range SfM Software Performance under Varying Capturing Conditions
    <https://data.mendeley.com/datasets/bzxk2n78s9/4>`_

    Original datasets are under the CC BY 4.0 license.

    For more details, see `Ivan Nikolov Datasets
    <https://github.com/pyvista/vtk-data/tree/master/Data/ivan-nikolov>`_

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
    >>> mesh = examples.download_plastic_vase()
    >>> mesh.plot()

    Return the statistics of the dataset.

    >>> mesh
    PolyData (...)
      N Cells:    3570967
      N Points:   1796805
      N Strips:   0
      X Bounds:   -1.364e+02, 1.929e+02
      Y Bounds:   -1.677e+02, 1.603e+02
      Z Bounds:   1.209e+02, 4.090e+02
      N Arrays:   0

    .. seealso::

        :ref:`Plastic Vase Dataset <plastic_vase_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_plastic_vase, load=load)


_dataset_plastic_vase = _SingleFileDownloadableDatasetLoader(
    'ivan-nikolov/plasticVase.zip',
    target_file='plasticVase.vtp',
)


def download_sea_vase(load=True):  # pragma: no cover
    """Download a scan of a sea vase created by Ivan Nikolov.

    The dataset was downloaded from `GGG-BenchmarkSfM: Dataset for Benchmarking
    Close-range SfM Software Performance under Varying Capturing Conditions
    <https://data.mendeley.com/datasets/bzxk2n78s9/4>`_

    Original datasets are under the CC BY 4.0 license.

    For more details, see `Ivan Nikolov Datasets
    <https://github.com/pyvista/vtk-data/tree/master/Data/ivan-nikolov>`_

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
    >>> mesh = examples.download_sea_vase()
    >>> mesh.plot()

    Return the statistics of the dataset.

    >>> mesh
    PolyData (...)
      N Cells:    3548473
      N Points:   1810012
      N Strips:   0
      X Bounds:   -1.666e+02, 1.465e+02
      Y Bounds:   -1.742e+02, 1.384e+02
      Z Bounds:   -1.500e+02, 2.992e+02
      N Arrays:   0

    .. seealso::

        :ref:`Sea Vase Dataset <sea_vase_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_sea_vase, load=load)


_dataset_sea_vase = _SingleFileDownloadableDatasetLoader(
    'ivan-nikolov/seaVase.zip',
    target_file='seaVase.vtp',
)


def download_dikhololo_night(load=True):  # pragma: no cover
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
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> gltf_file = examples.gltf.download_damaged_helmet()
    >>> texture = examples.download_dikhololo_night()
    >>> pl = pv.Plotter()
    >>> pl.import_gltf(gltf_file)
    >>> pl.set_environment_texture(texture)
    >>> pl.show()

    .. seealso::

        :ref:`Dikhololo Night Dataset <dikhololo_night_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_dikhololo_night, load=load)


def _dikhololo_night_load_func(texture):  # pragma: no cover
    texture.SetColorModeToDirectScalars()
    texture.SetMipmap(True)
    texture.SetInterpolate(True)
    return texture


_dataset_dikhololo_night = _SingleFileDownloadableDatasetLoader(
    'dikhololo_night_4k.hdr',
    read_func=read_texture,
)


def download_cad_model_case(load=True):  # pragma: no cover
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


def download_aero_bracket(load=True):  # pragma: no cover
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


def download_coil_magnetic_field(load=True):  # pragma: no cover
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
    ...         pv.Polygon((0, 0, z), radius=5, n_sides=100, fill=False)
    ...     )
    ...
    >>> coils = pv.MultiBlock(coils)
    >>> # plot the magnet field strength in the Z direction
    >>> scalars = np.abs(grid['B'][:, 2])
    >>> pl = pv.Plotter()
    >>> _ = pl.add_mesh(
    ...     coils, render_lines_as_tubes=True, line_width=5, color='w'
    ... )
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


def download_meshio_xdmf(load=True):  # pragma: no cover
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


def download_victorian_goblet_face_illusion(load=True):  # pragma: no cover
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
    >>> _ = plotter.add_floor('-x', color="black")
    >>> plotter.enable_parallel_projection()
    >>> plotter.show(cpos="yz")

    .. seealso::

        :ref:`Victorian Goblet Face Illusion Dataset <victorian_goblet_face_illusion_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_victorian_goblet_face_illusion, load=load)


_dataset_victorian_goblet_face_illusion = _SingleFileDownloadableDatasetLoader(
    'Victorian_Goblet_face_illusion/Vase.stl',
)


def download_reservoir(load=True):  # pragma: no cover
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


def _reservoir_load_func(grid):  # pragma: no cover
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
    read_func=pyvista.ExplicitStructuredGrid,
    load_func=_reservoir_load_func,
)


def download_whole_body_ct_male(load=True):  # pragma: no cover
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

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

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

    Get the CT image

    >>> ct_image = dataset['ct']
    >>> ct_image
    ImageData (...)
      N Cells:      55561506
      N Points:     56012800
      X Bounds:     0.000e+00, 4.785e+02
      Y Bounds:     0.000e+00, 4.785e+02
      Z Bounds:     0.000e+00, 8.190e+02
      Dimensions:   320, 320, 547
      Spacing:      1.500e+00, 1.500e+00, 1.500e+00
      N Arrays:     1

    Get the segmentation label names and show the first three

    >>> segmentations = dataset['segmentations']
    >>> label_names = segmentations.keys()
    >>> label_names[:3]
    ['adrenal_gland_left', 'adrenal_gland_right', 'aorta']

    Get the label map and show its data range

    >>> label_map = dataset['label_map']
    >>> label_map.get_data_range()
    (np.uint8(0), np.uint8(117))

    Create a surface mesh of the segmentation labels

    >>> labels_mesh = label_map.contour_labeled(smoothing=True)

    Plot the CT image and segmentation labels together.

    >>> pl = pv.Plotter()
    >>> _ = pl.add_volume(
    ...     ct_image,
    ...     cmap="bone",
    ...     opacity="sigmoid_9",
    ...     show_scalar_bar=False,
    ... )
    >>> _ = pl.add_mesh(labels_mesh, cmap='glasbey', show_scalar_bar=False)
    >>> pl.view_zx()
    >>> pl.camera.up = (0, 0, 1)
    >>> pl.camera.zoom(1.3)
    >>> pl.show()

    .. seealso::

        :ref:`Whole Body Ct Male Dataset <whole_body_ct_male_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Whole Body Ct Female Dataset <whole_body_ct_female_dataset>`
            Similar dataset of a female subject.

        :ref:`medical_dataset_gallery`
            Browse other medical datasets.

    """
    return _download_dataset(_dataset_whole_body_ct_male, load=load)


def _whole_body_ct_load_func(dataset):  # pragma: no cover
    # Process the dataset to create a label map from the segmentation masks

    segmentations = dataset['segmentations']

    # Create label map array from segmentation masks
    # Initialize array with background values (zeros)
    label_map_array = np.zeros((segmentations[0].n_points,), dtype=np.uint8)
    label_names = sorted(segmentations.keys())
    for i, name in enumerate(label_names):
        label_map_array[segmentations[name].active_scalars == 1] = i + 1

    # Add scalars to a new image
    label_map_image = segmentations[0].copy()
    label_map_image.clear_data()
    label_map_image['label_map'] = label_map_array

    # Add label map to dataset
    dataset['label_map'] = label_map_image

    return dataset


_dataset_whole_body_ct_male = _SingleFileDownloadableDatasetLoader(
    'whole_body_ct/s1397.zip',
    target_file='s1397',
    load_func=_whole_body_ct_load_func,
)


def download_whole_body_ct_female(load=True):  # pragma: no cover
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

    Parameters
    ----------
    load : bool, default: True
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.MultiBlock or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    Load the dataset

    >>> from pyvista import examples
    >>> import pyvista as pv
    >>> dataset = examples.download_whole_body_ct_female()

    Get the names of the dataset's blocks

    >>> dataset.keys()
    ['ct', 'segmentations', 'label_map']

    Get the CT image

    >>> ct_image = dataset['ct']
    >>> ct_image
    ImageData (...)
      N Cells:      55154462
      N Points:     55603200
      X Bounds:     0.000e+00, 4.785e+02
      Y Bounds:     0.000e+00, 4.785e+02
      Z Bounds:     0.000e+00, 8.130e+02
      Dimensions:   320, 320, 543
      Spacing:      1.500e+00, 1.500e+00, 1.500e+00
      N Arrays:     1

    Get the segmentation label names and show the first three

    >>> segmentations = dataset['segmentations']
    >>> label_names = segmentations.keys()
    >>> label_names[:3]
    ['adrenal_gland_left', 'adrenal_gland_right', 'aorta']

    Get the label map and show its data range

    >>> label_map = dataset['label_map']
    >>> label_map.get_data_range()
    (np.uint8(0), np.uint8(117))

    Create a surface mesh of the segmentation labels

    >>> labels_mesh = label_map.contour_labeled(smoothing=True)

    Plot the CT image and segmentation labels together.

    >>> pl = pv.Plotter()
    >>> _ = pl.add_volume(
    ...     ct_image,
    ...     cmap="bone",
    ...     opacity="sigmoid_7",
    ...     show_scalar_bar=False,
    ... )
    >>> _ = pl.add_mesh(labels_mesh, cmap='glasbey', show_scalar_bar=False)
    >>> pl.view_zx()
    >>> pl.camera.up = (0, 0, 1)
    >>> pl.camera.zoom(1.3)
    >>> pl.show()

    .. seealso::

        :ref:`Whole Body Ct Female Dataset <whole_body_ct_female_dataset>`
            See this dataset in the Dataset Gallery for more info.

        :ref:`Whole Body Ct Male Dataset <whole_body_ct_male_dataset>`
            Similar dataset of a male subject.

        :ref:`medical_dataset_gallery`
            Browse other medical datasets.

    """
    return _download_dataset(_dataset_whole_body_ct_female, load=load)


_dataset_whole_body_ct_female = _SingleFileDownloadableDatasetLoader(
    'whole_body_ct/s1380.zip',
    target_file='s1380',
    load_func=_whole_body_ct_load_func,
)


def download_room_cff(load=True):  # pragma: no cover
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
    >>> mesh.plot(cpos="xy", scalars="SV_T")

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


def download_m4_total_density(load=True):  # pragma: no cover
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
    >>> outline = pl.add_mesh(grid.outline(), color="black")
    >>> volume = pl.add_volume(grid)

    Add atoms and bonds to the plotter.

    >>> atoms = pl.add_mesh(poly.glyph(geom=pv.Sphere()), color="red")
    >>> bonds = pl.add_mesh(poly.tube(), color="white")

    >>> pl.show(cpos="zx")

    .. seealso::

        :ref:`M4 Total Density Dataset <m4_total_density_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_m4_total_density, load=load)


_dataset_m4_total_density = _SingleFileDownloadableDatasetLoader('m4_TotalDensity.cube')


def download_headsq(load=True):  # pragma: no cover
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
    >>> mesh.plot(cpos="xy")

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


def download_prism(load=True):  # pragma: no cover
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


def download_t3_grid_0(load=True):  # pragma: no cover
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


def download_caffeine(load=True):  # pragma: no cover
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
    >>> atoms = pl.add_mesh(
    ...     poly.glyph(geom=pv.Sphere(radius=0.1)), color="red"
    ... )
    >>> bonds = pl.add_mesh(poly.tube(radius=0.1), color="gray")
    >>> pl.show(cpos="xy")

    .. seealso::

        :ref:`Caffeine Dataset <caffeine_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_caffeine, load=load)


_dataset_caffeine = _SingleFileDownloadableDatasetLoader('caffeine.pdb')


def download_e07733s002i009(load=True):  # paragma: no cover
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


def download_particles(load=True):  # pragma: no cover
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


def download_prostar(load=True):  # pragma: no cover
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


def _prostar_files_func():  # pragma: no cover
    # Multiple files needed for read, but only one gets loaded
    prostar_cel = _DownloadableFile('prostar.cel')
    prostar_vrt = _SingleFileDownloadableDatasetLoader('prostar.vrt')
    return prostar_vrt, prostar_cel


_dataset_prostar = _MultiFileDownloadableDatasetLoader(_prostar_files_func)


def download_3gqp(load=True):  # pragma: no cover
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

        :ref:`3GQP Dataset <3gqp_dataset>`
            See this dataset in the Dataset Gallery for more info.

    """
    return _download_dataset(_dataset_3gqp, load=load)


_dataset_3gqp = _SingleFileDownloadableDatasetLoader('3GQP.pdb')
