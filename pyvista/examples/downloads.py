"""Downloadable datasets collected from various sources.

Once downloaded, these datasets are stored locally allowing for the
rapid reuse of these datasets.

Files are all hosted in https://github.com/pyvista/vtk-data/ and are downloaded
using the ``download_file`` function. If you add a file to the example data
repository, you should add a ``download-<dataset>`` method here which will
rendered on this page.

Examples
--------
>>> from pyvista import examples
>>> mesh = examples.download_saddle_surface()
>>> mesh.plot()

"""
import logging
import os
from pathlib import PureWindowsPath
import shutil
from typing import Union
import warnings

import numpy as np
import pooch
from pooch import Unzip
from pooch.utils import get_logger

import pyvista
from pyvista.core import _vtk_core as _vtk
from pyvista.core.errors import PyVistaDeprecationWarning, VTKVersionError
from pyvista.core.utilities.fileio import get_ext, read, read_texture
from pyvista.core.utilities.reader import DICOMReader

# disable pooch verbose logging
POOCH_LOGGER = get_logger()
POOCH_LOGGER.setLevel(logging.CRITICAL)


CACHE_VERSION = 3

# If available, a local vtk-data instance will be used for examples
if 'PYVISTA_VTK_DATA' in os.environ:  # pragma: no cover
    _path = os.environ['PYVISTA_VTK_DATA']

    if not os.path.basename(_path) == 'Data':
        # append 'Data' if user does not provide it
        _path = os.path.join(_path, 'Data')

    # pooch assumes this is a URL so we have to take care of this
    if not _path.endswith('/'):
        _path = _path + '/'
    SOURCE = _path
    _FILE_CACHE = True

else:
    SOURCE = "https://github.com/pyvista/vtk-data/raw/master/Data/"
    _FILE_CACHE = False

# allow user to override the local path
if 'PYVISTA_USERDATA_PATH' in os.environ:  # pragma: no cover
    if not os.path.isdir(os.environ['PYVISTA_USERDATA_PATH']):
        warnings.warn('Ignoring invalid {PYVISTA_USERDATA_PATH')
    else:
        USER_DATA_PATH = os.environ['PYVISTA_USERDATA_PATH']
else:
    # use default pooch path
    USER_DATA_PATH = str(pooch.os_cache(f'pyvista_{CACHE_VERSION}'))

    # provide helpful message if pooch path is inaccessible
    if not os.path.isdir(USER_DATA_PATH):  # pragma: no cover
        try:
            os.makedirs(USER_DATA_PATH, exist_ok=True)
            if not os.access(USER_DATA_PATH, os.W_OK):
                raise OSError
        except (PermissionError, OSError):
            # Warn, don't raise just in case there's an environment issue.
            warnings.warn(
                f'Unable to access {USER_DATA_PATH}. Manually specify the PyVista'
                'examples cache with the PYVISTA_USERDATA_PATH environment variable.'
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
    if not os.path.isfile(input_file):
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


def delete_downloads():
    """Delete all downloaded examples to free space or update the files.

    Examples
    --------
    Delete all local downloads.

    >>> from pyvista import examples
    >>> examples.delete_downloads()  # doctest:+SKIP

    """
    if os.path.isdir(USER_DATA_PATH):
        shutil.rmtree(USER_DATA_PATH)
    os.makedirs(USER_DATA_PATH)


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

    See :ref:`texture_example` for an example using this
    dataset.

    """
    return _download_and_read('masonry.bmp', texture=True, load=load)


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

    """
    return _download_and_read('usa_image.jpg', texture=True, load=load)


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

    See :ref:`texture_example` for an example using this
    dataset.

    """
    return _download_and_read('puppy.jpg', texture=True, load=load)


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

    """
    return _download_and_read('puppy.jpg', load=load)


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

    """
    return _download_and_read('usa.vtk', load=load)


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

    This dataset is used in the following examples:

    * :ref:`colormap_example`
    * :ref:`lighting_properties_example`
    * :ref:`plot_opacity_example`
    * :ref:`orbiting_example`
    * :ref:`plot_over_line_example`
    * :ref:`plotter_lighting_example`
    * :ref:`themes_example`

    """
    return _download_and_read('SainteHelens.dem', load=load)


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

    See Also
    --------
    download_bunny_coarse

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_bunny()
    >>> dataset.plot(cpos="xy")

    This dataset is used in the following examples:

    * :ref:`read_file_example`
    * :ref:`clip_with_surface_example`
    * :ref:`extract_edges_example`
    * :ref:`subdivide_example`
    * :ref:`silhouette_example`
    * :ref:`light_types_example`

    """
    return _download_and_read('bunny.ply', load=load)


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

    See Also
    --------
    download_bunny

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_bunny_coarse()
    >>> dataset.plot(cpos="xy")

    * :ref:`read_file_example`
    * :ref:`clip_with_surface_example`
    * :ref:`subdivide_example`

    """
    result = _download_and_read('Bunny.vtp', load=load)
    if load:
        result.verts = np.array([], dtype=np.int32)
    return result


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

    This dataset is used in the following examples:

    * :ref:`extract_edges_example`
    * :ref:`mesh_quality_example`
    * :ref:`rotate_example`
    * :ref:`linked_views_example`
    * :ref:`light_actors_example`

    """
    return _download_and_read('cow.vtp', load=load)


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

    """
    return _download_and_read('cowHead.vtp', load=load)


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

    """
    return _download_and_read('faults.vtk', load=load)


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

    """
    return _download_and_read('tensors.vtk', load=load)


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

    See :ref:`volume_rendering_example` for an example using this
    dataset.

    """
    download_file('HeadMRVolume.raw')
    return _download_and_read('HeadMRVolume.mhd', load=load)


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

    """
    download_file('head.vti')
    return _download_and_read('head.vti', load=load)


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

    See :ref:`volume_rendering_example` for an example using this
    dataset.

    """
    if not load:
        return (_download_and_read('bolt.slc', load=load), _download_and_read('nut.slc', load=load))
    blocks = pyvista.MultiBlock()
    blocks['bolt'] = _download_and_read('bolt.slc')
    blocks['nut'] = _download_and_read('nut.slc')
    return blocks


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

    """
    return _download_and_read('clown.facet', load=load)


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

    This dataset is used in the following examples:

    * :ref:`surface_normal_example`
    * :ref:`background_image_example`

    """
    return _download_and_read('EarthModels/ETOPO_10min_Ice.vtp', load=load)


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

    This dataset is used in the following examples:

    * :ref:`geodesic_example`
    * :ref:`background_image_example`

    """
    return _download_and_read('EarthModels/ETOPO_10min_Ice_only-land.vtp', load=load)


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

    """
    return _download_and_read('EarthModels/Coastlines_Los_Alamos.vtp', load=load)


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

    This dataset is used in the following examples:

    * :ref:`plot_opacity_example`
    * :ref:`volume_rendering_example`
    * :ref:`slider_bar_widget_example`

    """
    return _download_and_read('DICOM_KNEE.dcm', load=load)


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

    This dataset is used in the following examples:

    * :ref:`volume_rendering_example`
    * :ref:`slider_bar_widget_example`

    """
    return _download_and_read('vw_knee.slc', load=load)


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

    This dataset is used in the following examples:

    * :ref:`create_point_cloud`
    * :ref:`edl`

    """
    return _download_and_read('kafadar-lidar-interp.vtp', load=load)


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

    """
    return _download_and_read('mesh_fs8.exo', load=load)


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

    This dataset is used in the following examples:

    * :ref:`surface_normal_example`
    * :ref:`extract_edges_example`
    * :ref:`show_edges_example`
    * :ref:`edl`
    * :ref:`pbr_example`
    * :ref:`box_widget_example`

    """
    filename = _download_archive('nefertiti.ply.zip', target_file='nefertiti.ply')

    if not load:
        return filename
    return pyvista.read(filename)


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

    This dataset is used in the following examples:

    * :ref:`read_parallel_example`
    * :ref:`streamlines_example`
    * :ref:`integrate_example`

    """
    filename = _download_archive(
        'pvtu_blood_vessels/blood_vessels.zip', target_file='T0000000500.pvtu'
    )

    if not load:
        return filename
    mesh = pyvista.read(filename)
    mesh.set_active_vectors('velocity')
    return mesh


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

    """
    return _download_and_read('ironProt.vtk', load=load)


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

    """
    return _download_and_read('Tetrahedron.vtu', load=load)


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

    See :ref:`interpolate_example` for an example using this
    dataset.

    """
    return _download_and_read('InterpolatingOnSTL_final.stl', load=load)


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

    See :ref:`interpolate_example` for an example using this
    dataset.

    """
    saved_file = download_file('sparsePoints.txt')
    if not load:
        return saved_file
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

    See :ref:`voxelize_surface_mesh_example` for an example using this
    dataset.

    """
    return _download_and_read('fsu/footbones.ply', load=load)


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

    """
    return _download_and_read('fsu/stratocaster.ply', load=load)


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

    """
    return _download_and_read('QuadraticPyramid.vtu', load=load)


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

    """
    return _download_and_read('Pileated.jpg', load=load)


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

    """
    return _download_and_read('Pileated.jpg', texture=True, load=load)


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

    See :ref:`clip_with_plane_box_example` for an example using this
    dataset.

    """
    return _download_and_read('office.binary.vtk', load=load)


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

    """
    return _download_and_read('horsePoints.vtp', load=load)


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

    See :ref:`disabling_mesh_lighting_example` for an example using
    this dataset.

    """
    return _download_and_read('horse.vtp', load=load)


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

    """
    return _download_and_read('cake_easy.jpg', load=load)


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

    """
    return _download_and_read('cake_easy.jpg', texture=True, load=load)


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

    """
    return _download_and_read('RectilinearGrid.vtr', load=load)


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

    See :ref:`gaussian_smoothing_example` for an example using
    this dataset.

    """
    if zoom:
        return _download_and_read('Gourds.png', load=load)
    return _download_and_read('Gourds2.jpg', load=load)


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

    """
    if zoom:
        return _download_and_read('Gourds.png', texture=True, load=load)
    return _download_and_read('Gourds2.jpg', texture=True, load=load)


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

    """
    return _download_and_read('Gourds.pnm', load=load)


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

    """
    return _download_and_read('uGridEx.vtk', load=load)


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

    """
    return _download_and_read('k.vtk', load=load)


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

    See :ref:`cell_centers_example` for an example using
    this dataset.

    """
    return _download_and_read('a_grid.vtk', load=load)


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

    """
    return _download_and_read('polyline.vtk', load=load)


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

    See :ref:`read_file_example` for an example using
    this dataset.

    """
    return _download_and_read('42400-IDGH.stl', load=load)


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

    See :func:`download_frog_tissue` for segmentation labels associated
    with this dataset.

    See :ref:`volume_rendering_example` for an example using this dataset.

    """
    download_file('froggy/frog.zraw')
    return _download_and_read('froggy/frog.mhd', load=load)


def download_frog_tissue(load=True):  # pragma: no cover
    """Download frog tissue dataset.

    This dataset contains tissue segmentation labels for the frog dataset
    (see :func:`download_frog`).

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
    Load data

    >>> import numpy as np
    >>> import pyvista as pv
    >>> from pyvista import examples
    >>> data = examples.download_frog_tissue()

    Plot tissue labels as a volume

    First, define plotting parameters

    >>> # Configure colors / color bar
    >>> clim = data.get_data_range()  # Set color bar limits to match data
    >>> cmap = 'glasbey'  # Use a categorical colormap
    >>> categories = True  # Ensure n_colors matches number of labels
    >>> opacity = (
    ...     'foreground'  # Make foreground opaque, background transparent
    ... )
    >>> opacity_unit_distance = 1

    Set plotting resolution to half the image's spacing

    >>> res = np.array(data.spacing) / 2

    Define rendering parameters

    >>> mapper = 'gpu'
    >>> shade = True
    >>> ambient = 0.3
    >>> diffuse = 0.6
    >>> specular = 0.5
    >>> specular_power = 40

    Make and show plot

    >>> p = pv.Plotter()
    >>> _ = p.add_volume(
    ...     data,
    ...     clim=clim,
    ...     ambient=ambient,
    ...     shade=shade,
    ...     diffuse=diffuse,
    ...     specular=specular,
    ...     specular_power=specular_power,
    ...     mapper=mapper,
    ...     opacity=opacity,
    ...     opacity_unit_distance=opacity_unit_distance,
    ...     categories=categories,
    ...     cmap=cmap,
    ...     resolution=res,
    ... )
    >>> p.camera_position = 'yx'  # Set camera to provide a dorsal view
    >>> p.show()

    """
    download_file('froggy/frogtissue.zraw')
    return _download_and_read('froggy/frogtissue.mhd', load=load)


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

    See :ref:`volume_rendering_example` for an example using
    this dataset.

    """
    return _download_and_read('MetaIO/ChestCT-SHORT.mha', load=load)


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

    """
    return _download_and_read('avg152T1_RL_nifti.nii.gz', load=load)


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

    """
    return _download_and_read('prostate.img', load=load)


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

    """
    return _download_and_read('filledContours.vtp', load=load)


def download_doorman(load=True):  # pragma: no cover
    """Download doorman dataset.

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

    See :ref:`read_file_example` for an example using
    this dataset.

    """
    # TODO: download textures as well
    return _download_and_read('doorman/doorman.obj', load=load)


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

    """
    return _download_and_read('mug.e', load=load)


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

    """
    return _download_and_read('ObliqueCone.vtp', load=load)


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

    """
    return _download_and_read('emote.jpg', load=load)


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

    """
    return _download_and_read('emote.jpg', texture=True, load=load)


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

    This dataset is used in the following examples:

    * :ref:`read_file_example`
    * :ref:`cell_centers_example`

    """
    return _download_and_read('teapot.g', load=load)


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

    This dataset is used in the following examples:

    * :ref:`gaussian_smoothing_example`
    * :ref:`slice_example`
    * :ref:`depth_peeling_example`
    * :ref:`moving_isovalue_example`
    * :ref:`plane_widget_example`

    """
    return _download_and_read('brain.vtk', load=load)


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

    """
    return _download_and_read('StructuredGrid.vts', load=load)


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

    """
    return _download_and_read('SampleStructGrid.vtk', load=load)


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

    """
    return _download_and_read('trumpet.obj', load=load)


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

    See :ref:`decimate_example` for an example using
    this dataset.


    """
    # TODO: there is a texture with this
    return _download_and_read('fran_cut.vtk', load=load)


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

    """
    return _download_and_read('skybox-nz.jpg', load=load)


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

    """
    return _download_and_read('skybox-nz.jpg', texture=True, load=load)


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

    """
    return _download_and_read('Disc_BiQuadraticQuads_0_0.vtu', load=load)


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

    """
    return _download_and_read('honolulu.vtk', load=load)


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

    """
    return _download_and_read('motor.g', load=load)


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

    """
    dataset = _download_and_read('TriQuadraticHexahedron.vtu', load=load)
    if load:
        dataset.clear_data()
    return dataset


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
    >>> dataset.plot()

    """
    return _download_and_read('Human.vtp', load=load)


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

    """
    return _download_and_read('vtk.vtp', load=load)


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

    """
    return _download_and_read('spider.ply', load=load)


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

    This dataset is used in the following examples:

    * :ref:`glyph_example`
    * :ref:`gradients_example`
    * :ref:`streamlines_example`
    * :ref:`plane_widget_example`

    """
    mesh = _download_and_read('carotid.vtk', load=load)
    if not load:
        return mesh
    mesh.set_active_scalars('scalars')
    mesh.set_active_vectors('vectors')
    return mesh


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

    """
    return _download_and_read('blow.vtk', load=load)


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

    """
    return _download_and_read('shark.ply', load=load)


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

    This dataset is used in the following examples:

    * :ref:`floors_example`
    * :ref:`orbiting_example`
    * :ref:`silhouette_example`
    * :ref:`light_shadows_example`

    """
    return _download_and_read('dragon.ply', load=load)


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

    """
    return _download_and_read('Armadillo.ply', load=load)


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
    """
    return _download_and_read('gears.stl', load=load)


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

    """
    return _download_and_read('Torso.vtp', load=load)


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
    >>> from pyvista import examples
    >>> dataset = examples.download_kitchen()
    >>> dataset.streamlines(n_points=5).plot()

    This dataset is used in the following examples:

    * :ref:`plot_over_line_example`
    * :ref:`line_widget_example`

    """
    mesh = _download_and_read('kitchen.vtk', load=load)
    if not load:
        return mesh
    if not split:
        return mesh
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


def download_tetra_dc_mesh():  # pragma: no cover
    """Download two meshes defining an electrical inverse problem.

    This contains a high resolution forward modeled mesh and a coarse
    inverse modeled mesh.

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

    """
    fnames = _download_archive('dc-inversion.zip')
    fwd = pyvista.read(file_from_files('mesh-forward.vtu', fnames))
    fwd.set_active_scalars('Resistivity(log10)-fwd')
    inv = pyvista.read(file_from_files('mesh-inverse.vtu', fnames))
    inv.set_active_scalars('Resistivity(log10)')
    return pyvista.MultiBlock({'forward': fwd, 'inverse': inv})


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

    See :ref:`plot_opacity_example` for an example using this dataset.

    """
    return _download_and_read("model_with_variance.vtu", load=load)


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

    See :ref:`interpolate_example` for an example using this dataset.

    """
    return _download_and_read("probes.vtp", load=load)


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

    """
    return _download_and_read("carburetor.ply", load=load)


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

    """
    return _download_and_read('turbineblade.ply', load=load)


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

    See :ref:`connectivity_example` for an example using this dataset.

    """
    return _download_and_read('pine_root.tri', load=load)


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

    This dataset is used in the following examples:

    * :ref:`terrain_following_mesh_example`
    * :ref:`topo_map_example`

    """
    return _download_and_read('Ruapehu_mag_dem_15m_NZTM.vtk', load=load)


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

    See :ref:`topo_map_example` for an example using this dataset.

    """
    return _download_and_read('BJ34_GeoTifv1-04_crater_clip.tif', texture=True, load=load)


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

    """
    return _download_and_read('dolfin_fine.xml', file_format="dolfin-xml", load=load)


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

    See :ref:`volume_rendering_example` for an example using this dataset.

    """
    volume = _download_and_read("damavand-volcano.vtk", load=load)
    if not load:
        return volume
    volume.rename_array("None", "data")
    return volume


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

    """
    return _download_and_read('250.vtk', load=load)


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

    This dataset is used in the following examples:

    * :ref:`contouring_example`
    * :ref:`resampling_example`
    * :ref:`orthogonal_slices_example`

    """
    filename = _download_and_read('embryo.slc', load=False)
    if load:
        # cleanup artifact
        dataset = pyvista.read(filename)
        mask = dataset['SLCImage'] == 255
        dataset['SLCImage'][mask] = 0
        return dataset
    else:
        return filename


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

    See :ref:`antarctica_example` for an example using this dataset.

    """
    return _download_and_read("antarctica_velocity.vtp", load=load)


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

    See :ref:`depth_peeling_example` for an example using this dataset.

    """
    return _download_and_read("room_surface_mesh.obj", load=load)


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

    """
    return _download_and_read("beach.nrrd", load=load)


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

    See :ref:`texture_example` for an example using this dataset.

    """
    return _download_and_read("alphachannel.png", texture=True, load=load)


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

    """
    return _download_and_read("vtk.png", texture=True, load=load)


def download_sky_box_cube_map():  # pragma: no cover
    """Download a skybox cube map texture.

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

    See :ref:`pbr_example` for an example using this dataset.

    """
    prefix = 'skybox2-'
    sets = ['posx', 'negx', 'posy', 'negy', 'posz', 'negz']
    images = [prefix + suffix + '.jpg' for suffix in sets]
    for image in images:
        download_file(image)

    return pyvista.cubemap(str(FETCHER.path), prefix)


def download_cubemap_park():  # pragma: no cover
    """Download a cubemap of a park.

    Downloaded from http://www.humus.name/index.php?page=Textures
    by David Eck, and converted to a smaller 512x512 size for use
    with WebGL in his free, on-line textbook at
    http://math.hws.edu/graphicsbook

    This work is licensed under a Creative Commons Attribution 3.0 Unported
    License.

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

    """
    fnames = download_file('cubemap_park/cubemap_park.zip')
    return pyvista.cubemap(os.path.dirname(fnames[0]))


def download_cubemap_space_4k():  # pragma: no cover
    """Download the 4k space cubemap.

    This cubemap was generated by downloading the 4k image from: `Deep Star
    Maps 2020 <https://svs.gsfc.nasa.gov/4851>`_ and converting it using
    https://jaxry.github.io/panorama-to-cubemap/

    See `vtk-data/cubemap_space
    <https://github.com/pyvista/vtk-data/tree/master/Data/cubemap_space#readme>`_
    for more details.

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

    """
    fnames = download_file('cubemap_space/4k.zip')
    return pyvista.cubemap(os.path.dirname(fnames[0]))


def download_cubemap_space_16k():  # pragma: no cover
    """Download the 16k space cubemap.

    This cubemap was generated by downloading the 16k image from: `Deep Star
    Maps 2020 <https://svs.gsfc.nasa.gov/4851>`_ and converting it using
    https://jaxry.github.io/panorama-to-cubemap/

    See `vtk-data/cubemap_space
    <https://github.com/pyvista/vtk-data/tree/master/Data/cubemap_space#readme>`_ for
    more details.

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

    """
    fnames = download_file('cubemap_space/16k.zip')
    return pyvista.cubemap(os.path.dirname(fnames[0]))


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

    """
    filename = _download_archive('EnSight.zip', 'foam_case_0_0_0_0.case')
    if not load:
        return filename
    return pyvista.read(filename)


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

    See :ref:`create_draped_surf_example` for an example using this dataset.

    """
    saved_file = download_file("gpr-example/data.npy")
    if not load:
        return saved_file
    return np.load(saved_file)


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

    See :ref:`create_draped_surf_example` for an example using this dataset.

    """
    saved_file = download_file("gpr-example/path.txt")
    if not load:
        return saved_file
    path = np.loadtxt(saved_file, skiprows=1)
    return pyvista.PolyData(path)


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

    """
    return _download_and_read('woman.stl', load=load)


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

    """
    return _download_and_read('lobster.ply', load=load)


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

    """
    return _download_and_read('man_face.stl', load=load)


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

    """
    return _download_and_read('urn.stl', load=load)


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

    """
    return _download_and_read('pepper.ply', load=load)


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

    """
    return _download_and_read('drill.obj', load=load)


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

    """
    return _download_and_read('tigerfighter.obj', load=load)


def download_mars_jpg():
    """Download and return the path of ``'mars.jpg'``.

    Returns
    -------
    str
        Filename of the JPEG.
    """
    # Deprecated on v0.37.0, estimated removal on v0.40.0
    warnings.warn(
        "examples.download_mars_jpg is deprecated.  Use examples.planets.download_mars_surface with"
        " load=False",
        PyVistaDeprecationWarning,
    )
    return pyvista.examples.planets.download_mars_surface(load=False)


def download_stars_jpg():
    """Download and return the path of ``'stars.jpg'``.

    Returns
    -------
    str
        Filename of the JPEG.
    """
    # Deprecated on v0.37.0, estimated removal on v0.40.0
    warnings.warn(
        "examples.download_stars_jpg is deprecated.  Use"
        " examples.planets.download_stars_sky_background with load=False",
        PyVistaDeprecationWarning,
    )
    return pyvista.examples.planets.download_stars_sky_background(load=False)


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

    """
    return _download_and_read('notch_stress.vtk', load=load)


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

    """
    return _download_and_read('notch_disp.vtu', load=load)


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

    See :ref:`pbr_example` for an example using this dataset.

    """
    return _download_and_read('louis.ply', load=load)


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

    See :ref:`2d_streamlines_example` for an example using this dataset.

    """
    filename = download_file('EnSight/CylinderCrossflow/cylinder_Re35.case')
    download_file('EnSight/CylinderCrossflow/cylinder_Re35.geo')
    download_file('EnSight/CylinderCrossflow/cylinder_Re35.scl1')
    download_file('EnSight/CylinderCrossflow/cylinder_Re35.scl2')
    download_file('EnSight/CylinderCrossflow/cylinder_Re35.vel')
    if not load:
        return filename
    return pyvista.read(filename)


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

    See :ref:`reader_example` for an example using this dataset.

    """
    filename = download_file('EnSight/naca.bin.case')
    download_file('EnSight/naca.gold.bin.DENS_1')
    download_file('EnSight/naca.gold.bin.DENS_3')
    download_file('EnSight/naca.gold.bin.geo')
    if not load:
        return filename
    return pyvista.read(filename)


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

    See :ref:`reader_example` for an example using this dataset.

    """
    filename = _download_archive('PVD/wavy.zip', 'unzip/wavy.pvd')
    if not load:
        return filename
    return pyvista.PVDReader(filename).read()


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

    """
    filename = _download_archive(
        'PVD/paraview/singleSphereAnimation.zip', 'singleSphereAnimation.pvd'
    )
    if not load:
        return filename
    return pyvista.PVDReader(filename).read()


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

    """
    filename = _download_archive(
        'PVD/paraview/dualSphereAnimation.zip',
        'dualSphereAnimation.pvd',
    )

    if not load:
        return filename
    return pyvista.PVDReader(filename).read()


def download_osmnx_graph():  # pragma: no cover
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

    Returns
    -------
    networkx.classes.multidigraph.MultiDiGraph
        An osmnx graph of the streets of Holzgerlingen, Germany.

    Examples
    --------
    >>> from pyvista import examples
    >>> graph = examples.download_osmnx_graph()  # doctest:+SKIP

    See :ref:`open_street_map_example` for a full example using this dataset.

    """
    import pickle

    try:
        import osmnx  # noqa
    except ImportError:
        raise ImportError('Install `osmnx` to use this example')

    filename = download_file('osmnx_graph.p')
    return pickle.load(open(filename, 'rb'))


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

    See :ref:`openfoam_example` for a full example using this dataset.

    """
    filename = _download_archive('OpenFOAM.zip', target_file='cavity/case.foam')
    if not load:
        return filename
    return pyvista.OpenFOAMReader(filename).read()


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

    See :ref:`openfoam_tubes_example` for a full example using this dataset.

    """
    filename = _download_archive(
        'fvm/turbo_incompressible/Turbo-Incompressible_3-Run_1-SOLUTION_FIELDS.zip',
        target_file='case.foam',
    )
    if not load:
        return filename
    reader = pyvista.OpenFOAMReader(filename)
    reader.set_active_time_value(1000)
    return reader.read()


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

    See :ref:`jupyter_plotting` for another example using this dataset.

    """
    return _download_and_read('lucy.ply', load=load)


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

    See :ref:`pump_bracket_example` for a full example using this dataset.

    """
    filename = _download_archive(
        'fea/pump_bracket/pump_bracket.zip',
        'pump_bracket.vtk',
    )
    if load:
        return pyvista.read(filename)
    return filename


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

    Show the type and bounds of the datasets.

    See :ref:`openfoam_cooling_example` for a full example using this dataset.

    """
    fnames = _download_archive('fvm/cooling_electronics/datasets.zip')
    if load:
        # return the structure dataset first
        if fnames[1].endswith('structure.vtp'):
            fnames = fnames[::-1]
        return pyvista.read(fnames[0]), pyvista.read(fnames[1])
    return fnames


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
    pyvista.PolyData, str, or List[str]
        The example ParaView can DataSet or file path(s).

    Examples
    --------
    Plot the can dataset.

    >>> from pyvista import examples
    >>> import pyvista as pv
    >>> dataset = examples.download_can()  # doctest:+SKIP
    >>> dataset.plot(scalars='VEL', smooth_shading=True)  # doctest:+SKIP

    """
    if pyvista.vtk_version_info > (9, 1):  # pragma: no cover
        raise VTKVersionError(
            'This example file is deprecated for VTK v9.2.0 and newer. '
            'Use `download_can_crushed_hdf` instead.'
        )

    can_0 = _download_and_read('hdf/can_0.hdf', load=load)
    if partial:
        return can_0

    cans = [
        can_0,
        _download_and_read('hdf/can_1.hdf', load=load),
        _download_and_read('hdf/can_2.hdf', load=load),
    ]

    if load:
        return pyvista.merge(cans)
    return cans


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

    """
    return _download_and_read('hdf/can-vtu.hdf', load=load)


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

    """
    filename = download_file('cgns/sqnz_s.adf.cgns')
    if not load:
        return filename
    return pyvista.get_reader(filename).read()


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

    """
    filename = download_file('tecplot_ascii.dat')
    if not load:
        return filename
    return pyvista.get_reader(filename).read()


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

    """
    filename = download_file('cgns/multi.cgns')
    if not load:
        return filename
    reader = pyvista.get_reader(filename)

    # disable reading the boundary patch. As of VTK 9.1.0 this generates
    # messages like "Skipping BC_t node: BC_t type 'BCFarfield' not supported
    # yet."
    reader.load_boundary_patch = False
    return reader.read()


def download_dicom_stack(load: bool = True) -> Union[pyvista.ImageData, str]:  # pragma: no cover
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

    """
    fnames = _download_archive('DICOM_Stack/data.zip')
    path = os.path.dirname(fnames[0])
    if load:
        reader = DICOMReader(path)
        return reader.read()
    return path


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

    """
    return _download_and_read("parched_canal_4k.hdr", texture=True, load=load)


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

    """
    return _download_and_read("cellsnd.ascii.inp", load=load)


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

    See :ref:`image_fft_example` for a full example using this dataset.

    """
    return _download_and_read('moonlanding.png', load=load)


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

    """
    return _download_and_read('AngularSector.vtk', load=load)


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

    """
    return _download_and_read('AOI.Damavand.32639.vtp', load=load)


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

    """
    return _download_and_read('lethe/result_particles.20000.0000.vtu', load=load)


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

    """
    return _download_and_read('gifs/sample.gif', load=load)


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

    See the :ref:`plotting_point_clouds` for a full example using this dataset.

    """
    filename = download_file('point-clouds/findus23/halo_low_res.npy')

    if load:
        return pyvista.PointSet(np.load(filename))
    return filename


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

    See the :ref:`plotting_point_clouds` for more details on how to plot point
    clouds.

    """
    filename = download_file('point-clouds/findus23/halo_high_res.npy')

    if load:
        return pyvista.PointSet(np.load(filename))
    return filename


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

    See the :ref:`plotting_point_clouds` for more details on how to plot point
    clouds.

    """
    return _download_and_read('point-clouds/hyg-database/stars.vtp', load=load)


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

    """
    return _download_and_read('fea/kiefer/dataset.vtu', load=load)


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

    """
    filename = _download_archive(
        'fea/hertzian_contact_cylinder/Hertzian_cylinder_on_plate.zip',
        target_file='bfac9fd1-e982-4825-9a95-9e5d8c5b4d3e_result_1.pvtu',
    )
    if load:
        return pyvista.read(filename)
    return filename


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


    """
    filename = _download_archive(
        'ivan-nikolov/blackVase.zip',
        'blackVase.vtp',
    )
    if load:
        return pyvista.read(filename)
    return filename


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

    """
    filename = _download_archive(
        'ivan-nikolov/Angel.zip',
        'Angel.vtp',
    )
    if load:
        return pyvista.read(filename)
    return filename


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

    """
    filename = _download_archive(
        'ivan-nikolov/birdBath.zip',
        'birdBath.vtp',
    )
    if load:
        return pyvista.read(filename)
    return filename


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

    """
    filename = _download_archive(
        'ivan-nikolov/owl.zip',
        'owl.vtp',
    )
    if load:
        return pyvista.read(filename)
    return filename


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

    """
    filename = _download_archive(
        'ivan-nikolov/plasticVase.zip',
        'plasticVase.vtp',
    )
    if load:
        return pyvista.read(filename)
    return filename


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

    """
    filename = _download_archive(
        'ivan-nikolov/seaVase.zip',
        'seaVase.vtp',
    )
    if load:
        return pyvista.read(filename)
    return filename


def download_dikhololo_night():  # pragma: no cover
    """Download and read the dikholo night hdr texture example.

    Files hosted at https://polyhaven.com/

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

    """
    texture = _download_and_read('dikhololo_night_4k.hdr', texture=True)
    texture.SetColorModeToDirectScalars()
    texture.SetMipmap(True)
    texture.SetInterpolate(True)
    return texture


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

    """
    return _download_and_read('cad/4947746/Vented_Rear_Case_With_Pi_Supports.vtp', load=load)


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

    """
    return _download_and_read('fea/aero_bracket/aero_bracket.vtu', load=load)


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
    ...     max_time=180,
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

    See the :ref:`magnetic_fields_example` for more details on how to plot with
    this dataset.

    """
    return _download_and_read('magpylib/coil_field.vti', load=load)


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

    """
    _ = download_file("meshio/out.h5")
    return _download_and_read("meshio/out.xdmf", load=load)
