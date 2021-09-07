"""Downloadable datasets collected from various sources.

Once downloaded, these datasets are stored locally allowing for the
rapid reuse of these datasets.

Examples
--------
>>> from pyvista import examples
>>> mesh = examples.download_saddle_surface()
>>> mesh.plot()

"""

from urllib.request import urlretrieve
from functools import partial
import os
import shutil
import sys
import zipfile

import numpy as np

import pyvista
from pyvista import _vtk


def _check_examples_path():
    """Check if the examples path exists."""
    if not pyvista.EXAMPLES_PATH:
        raise FileNotFoundError('EXAMPLES_PATH does not exist.  Try setting the '
                                'environment variable `PYVISTA_USERDATA_PATH` '
                                'to a writable path and restarting python')


def delete_downloads():
    """Delete all downloaded examples to free space or update the files.

    Returns
    -------
    bool
        Returns ``True``.

    Examples
    --------
    Delete all local downloads.

    >>> from pyvista import examples
    >>> examples.delete_downloads()  # doctest:+SKIP
    True

    """
    _check_examples_path()
    shutil.rmtree(pyvista.EXAMPLES_PATH)
    os.makedirs(pyvista.EXAMPLES_PATH)
    return True


def _decompress(filename):
    _check_examples_path()
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall(pyvista.EXAMPLES_PATH)
    return zip_ref.close()


def _get_vtk_file_url(filename):
    return f'https://github.com/pyvista/vtk-data/raw/master/Data/{filename}'


def _http_request(url):
    return urlretrieve(url)


def _repo_file_request(repo_path, filename):
    return os.path.join(repo_path, 'Data', filename), None


def _retrieve_file(retriever, filename):
    """Retrieve file and cache it in pyvsita.EXAMPLES_PATH.

    Parameters
    ----------
    retriever : str or callable
        If str, it is treated as a url.
        If callable, the function must take no arguments and must
        return a tuple like (file_path, resp), where file_path is
        the path to the file to use.
    filename : str
        The name of the file.
    """
    _check_examples_path()
    # First check if file has already been downloaded
    local_path = os.path.join(pyvista.EXAMPLES_PATH, os.path.basename(filename))
    local_path_no_zip = local_path.replace('.zip', '')
    if os.path.isfile(local_path_no_zip) or os.path.isdir(local_path_no_zip):
        return local_path_no_zip, None
    if isinstance(retriever, str):
        retriever = partial(_http_request, retriever)
    saved_file, resp = retriever()
    # new_name = saved_file.replace(os.path.basename(saved_file), os.path.basename(filename))
    # Make sure folder exists!
    if not os.path.isdir(os.path.dirname((local_path))):
        os.makedirs(os.path.dirname((local_path)))
    if pyvista.VTK_DATA_PATH is None:
        shutil.move(saved_file, local_path)
    else:
        if os.path.isdir(saved_file):
            shutil.copytree(saved_file, local_path)
        else:
            shutil.copy(saved_file, local_path)
    if pyvista.get_ext(local_path) in ['.zip']:
        _decompress(local_path)
        local_path = local_path[:-4]
    return local_path, resp


def _download_file(filename):
    if pyvista.VTK_DATA_PATH is None:
        url = _get_vtk_file_url(filename)
        retriever = partial(_http_request, url)
    else:
        if not os.path.isdir(pyvista.VTK_DATA_PATH):
            raise FileNotFoundError(f'VTK data repository path does not exist at:\n\n{pyvista.VTK_DATA_PATH}')
        if not os.path.isdir(os.path.join(pyvista.VTK_DATA_PATH, 'Data')):
            raise FileNotFoundError(f'VTK data repository does not have "Data" folder at:\n\n{pyvista.VTK_DATA_PATH}')
        retriever = partial(_repo_file_request, pyvista.VTK_DATA_PATH, filename)
    return _retrieve_file(retriever, filename)


def _download_and_read(filename, texture=False, file_format=None, load=True):
    saved_file, _ = _download_file(filename)
    if not load:
        return saved_file
    if texture:
        return pyvista.read_texture(saved_file)
    return pyvista.read(saved_file, file_format=file_format)


###############################################################################

def download_masonry_texture(load=True):  # pragma: no cover
    """Download masonry texture.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.DataSet or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_masonry_texture()  # doctest:+SKIP

    See :ref:`ref_texture_example` for an example using this
    dataset.

    """
    return _download_and_read('masonry.bmp', texture=True, load=load)


def download_usa_texture(load=True):  # pragma: no cover
    """Download usa texture.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.DataSet or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_usa_texture()  # doctest:+SKIP

    """
    return _download_and_read('usa_image.jpg', texture=True, load=load)


def download_puppy_texture(load=True):  # pragma: no cover
    """Download puppy texture.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.DataSet or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_puppy_texture()  # doctest:+SKIP

    See :ref:`ref_texture_example` for an example using this
    dataset.

    """
    return _download_and_read('puppy.jpg', texture=True, load=load)


def download_puppy(load=True):  # pragma: no cover
    """Download puppy dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UniformGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_puppy()  # doctest:+SKIP
    >>> dataset  # doctest:+SKIP
    UniformGrid (0x7fa0c15079a0)
    N Cells:	1917201
    N Points:	1920000
    X Bounds:	0.000e+00, 1.599e+03
    Y Bounds:	0.000e+00, 1.199e+03
    Z Bounds:	0.000e+00, 0.000e+00
    Dimensions:	1600, 1200, 1
    Spacing:	1.000e+00, 1.000e+00, 1.000e+00
    N Arrays:	1


    """
    return _download_and_read('puppy.jpg', load=load)


def download_usa(load=True):  # pragma: no cover
    """Download usa dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.DataSet or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_usa()  # doctest:+SKIP

    """
    return _download_and_read('usa.vtk', load=load)


def download_st_helens(load=True):  # pragma: no cover
    """Download Saint Helens dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UniformGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_st_helens()  # doctest:+SKIP

    This dataset is used in the following examples:

    * :ref:`colormap_example`
    * :ref:`ref_lighting_properties_example`
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
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    See Also
    --------
    download_bunny_coarse

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_bunny()  # doctest:+SKIP

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
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    See Also
    --------
    download_bunny

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_bunny_coarse()  # doctest:+SKIP

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
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_cow()  # doctest:+SKIP

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
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_cow_head()  # doctest:+SKIP

    """
    return _download_and_read('cowHead.vtp', load=load)


def download_faults(load=True):  # pragma: no cover
    """Download faults dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_faults()  # doctest:+SKIP

    """
    return _download_and_read('faults.vtk', load=load)


def download_tensors(load=True):  # pragma: no cover
    """Download tensors dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_tensors()  # doctest:+SKIP

    """
    return _download_and_read('tensors.vtk', load=load)


def download_head(load=True):  # pragma: no cover
    """Download head dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UniformGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_head()  # doctest:+SKIP

    See :ref:`volume_rendering_example` for an example using this
    dataset.

    """
    _download_file('HeadMRVolume.raw')
    return _download_and_read('HeadMRVolume.mhd', load=load)


def download_bolt_nut(load=True):  # pragma: no cover
    """Download bolt nut dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.MultiBlock or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_bolt_nut()  # doctest:+SKIP

    See :ref:`volume_rendering_example` for an example using this
    dataset.

    """
    if not load:
        return (
            _download_and_read('bolt.slc', load=load),
            _download_and_read('nut.slc', load=load)
        )
    blocks = pyvista.MultiBlock()
    blocks['bolt'] = _download_and_read('bolt.slc')
    blocks['nut'] = _download_and_read('nut.slc')
    return blocks


def download_clown(load=True):  # pragma: no cover
    """Download clown dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_clown()  # doctest:+SKIP

    """
    return _download_and_read('clown.facet', load=load)


def download_topo_global(load=True):  # pragma: no cover
    """Download topo dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_topo_global()  # doctest:+SKIP

    This dataset is used in the following examples:

    * :ref:`surface_normal_example`
    * :ref:`background_image_example`

    """
    return _download_and_read('EarthModels/ETOPO_10min_Ice.vtp', load=load)


def download_topo_land(load=True):  # pragma: no cover
    """Download topo land dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_topo_land()  # doctest:+SKIP

    This dataset is used in the following examples:

    * :ref:`geodesic_example`
    * :ref:`background_image_example`

    """
    return _download_and_read('EarthModels/ETOPO_10min_Ice_only-land.vtp', load=load)


def download_coastlines(load=True):  # pragma: no cover
    """Download coastlines dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_coastlines()  # doctest:+SKIP

    """
    return _download_and_read('EarthModels/Coastlines_Los_Alamos.vtp', load=load)


def download_knee(load=True):  # pragma: no cover
    """Download knee dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UniformGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_knee()  # doctest:+SKIP

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
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UniformGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_knee_full()  # doctest:+SKIP

    This dataset is used in the following examples:

    * :ref:`volume_rendering_example`
    * :ref:`slider_bar_widget_example`

    """
    return _download_and_read('vw_knee.slc', load=load)


def download_lidar(load=True):  # pragma: no cover
    """Download lidar dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_lidar()  # doctest:+SKIP

    This dataset is used in the following examples:

    * :ref:`create_point_cloud`
    * :ref:`ref_edl`

    """
    return _download_and_read('kafadar-lidar-interp.vtp', load=load)


def download_exodus(load=True):  # pragma: no cover
    """Sample ExodusII data file.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.MultiBlock or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_exodus()  # doctest:+SKIP

    """
    return _download_and_read('mesh_fs8.exo', load=load)


def download_nefertiti(load=True):  # pragma: no cover
    """Download mesh of Queen Nefertiti.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_nefertiti()  # doctest:+SKIP

    This dataset is used in the following examples:

    * :ref:`surface_normal_example`
    * :ref:`extract_edges_example`
    * :ref:`show_edges_example`
    * :ref:`ref_edl`
    * :ref:`pbr_example`
    * :ref:`box_widget_example`

    """
    return _download_and_read('nefertiti.ply.zip', load=load)


def download_blood_vessels(load=True):  # pragma: no cover
    """Download data representing the bifurcation of blood vessels.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_blood_vessels()  # doctest:+SKIP

    This dataset is used in the following examples:

    * :ref:`read_parallel_example`
    * :ref:`streamlines_example`

    """
    local_path, _ = _download_file('pvtu_blood_vessels/blood_vessels.zip')
    filename = os.path.join(local_path, 'T0000000500.pvtu')
    if not load:
        return filename
    mesh = pyvista.read(filename)
    mesh.set_active_vectors('velocity')
    return mesh


def download_iron_protein(load=True):  # pragma: no cover
    """Download iron protein dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UniformGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_iron_protein()  # doctest:+SKIP

    """
    return _download_and_read('ironProt.vtk', load=load)


def download_tetrahedron(load=True):  # pragma: no cover
    """Download tetrahedron dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_tetrahedron()  # doctest:+SKIP

    """
    return _download_and_read('Tetrahedron.vtu', load=load)


def download_saddle_surface(load=True):  # pragma: no cover
    """Download saddle surface dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_saddle_surface()  # doctest:+SKIP

    See :ref:`interpolate_example` for an example using this
    dataset.

    """
    return _download_and_read('InterpolatingOnSTL_final.stl', load=load)


def download_sparse_points(load=True):  # pragma: no cover
    """Download sparse points data.

    Used with :func:`download_saddle_surface`.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_sparse_points()  # doctest:+SKIP

    See :ref:`interpolate_example` for an example using this
    dataset.

    """
    saved_file, _ = _download_file('sparsePoints.txt')
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
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_foot_bones()  # doctest:+SKIP

    See :ref:`voxelize_surface_mesh_example` for an example using this
    dataset.

    """
    return _download_and_read('fsu/footbones.ply', load=load)


def download_guitar(load=True):  # pragma: no cover
    """Download guitar dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData} or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_guitar()  # doctest:+SKIP

    """
    return _download_and_read('fsu/stratocaster.ply', load=load)


def download_quadratic_pyramid(load=True):  # pragma: no cover
    """Download quadratic pyramid dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_quadratic_pyramid()  # doctest:+SKIP

    """
    return _download_and_read('QuadraticPyramid.vtu', load=load)


def download_bird(load=True):  # pragma: no cover
    """Download bird dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UniformGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_bird()  # doctest:+SKIP

    """
    return _download_and_read('Pileated.jpg', load=load)


def download_bird_texture(load=True):  # pragma: no cover
    """Download bird texture.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.Texture or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_bird_texture()  # doctest:+SKIP

    """
    return _download_and_read('Pileated.jpg', texture=True, load=load)


def download_office(load=True):  # pragma: no cover
    """Download office dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.StructuredGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_office()  # doctest:+SKIP

    See :ref:`clip_with_plane_box_example` for an example using this
    dataset.

    """
    return _download_and_read('office.binary.vtk', load=load)


def download_horse_points(load=True):  # pragma: no cover
    """Download horse points dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_horse_points()  # doctest:+SKIP

    """
    return _download_and_read('horsePoints.vtp', load=load)


def download_horse(load=True):  # pragma: no cover
    """Download horse dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.DataSet or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_horse()  # doctest:+SKIP

    See :ref:`disabling_mesh_lighting_example` for an example using
    this dataset.

    """
    return _download_and_read('horse.vtp', load=load)


def download_cake_easy(load=True):  # pragma: no cover
    """Download cake dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UniformGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_cake_easy()  # doctest:+SKIP

    """
    return _download_and_read('cake_easy.jpg', load=load)


def download_cake_easy_texture(load=True):  # pragma: no cover
    """Download cake texture.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.Texture or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_cake_easy_texture()  # doctest:+SKIP

    """
    return _download_and_read('cake_easy.jpg', texture=True, load=load)


def download_rectilinear_grid(load=True):  # pragma: no cover
    """Download rectilinear grid dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.RectilinearGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_rectilinear_grid()  # doctest:+SKIP

    """
    return _download_and_read('RectilinearGrid.vtr', load=load)


def download_gourds(zoom=False, load=True):  # pragma: no cover
    """Download gourds dataset.

    Parameters
    ----------
    zoom : bool, optional
        When ``True``, return the zoomed picture of the gourds.

    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UniformGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_gourds()  # doctest:+SKIP

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
    zoom : bool, optional
        When ``True``, return the zoomed picture of the gourds.

    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.DataSet or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_gourds_texture()  # doctest:+SKIP

    """
    if zoom:
        return _download_and_read('Gourds.png', texture=True, load=load)
    return _download_and_read('Gourds2.jpg', texture=True, load=load)


def download_unstructured_grid(load=True):  # pragma: no cover
    """Download unstructured grid dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_unstructured_grid()  # doctest:+SKIP

    """
    return _download_and_read('uGridEx.vtk', load=load)


def download_letter_k(load=True):  # pragma: no cover
    """Download letter k dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_letter_k()  # doctest:+SKIP

    """
    return _download_and_read('k.vtk', load=load)


def download_letter_a(load=True):  # pragma: no cover
    """Download letter a dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_letter_a()  # doctest:+SKIP

    See :ref:`cell_centers_example` for an example using
    this dataset.

    """
    return _download_and_read('a_grid.vtk', load=load)


def download_poly_line(load=True):  # pragma: no cover
    """Download polyline dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_poly_line()  # doctest:+SKIP

    """
    return _download_and_read('polyline.vtk', load=load)


def download_cad_model(load=True):  # pragma: no cover
    """Download cad dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_cad_model()  # doctest:+SKIP

    See :ref:`read_file_example` for an example using
    this dataset.

    """
    return _download_and_read('42400-IDGH.stl', load=load)


def download_frog(load=True):  # pragma: no cover
    """Download frog dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UniformGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_frog()  # doctest:+SKIP

    See :ref:`volume_rendering_example` for an example using
    this dataset.

    """
    # TODO: there are other files with this
    _download_file('froggy/frog.zraw')
    return _download_and_read('froggy/frog.mhd', load=load)


def download_prostate(load=True):  # pragma: no cover
    """Download prostate dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UniformGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_prostate()  # doctest:+SKIP

    """
    return _download_and_read('prostate.img', load=load)


def download_filled_contours(load=True):  # pragma: no cover
    """Download filled contours dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_filled_contours()  # doctest:+SKIP

    """
    return _download_and_read('filledContours.vtp', load=load)


def download_doorman(load=True):  # pragma: no cover
    """Download doorman dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_doorman()  # doctest:+SKIP

    See :ref:`read_file_example` for an example using
    this dataset.

    """
    # TODO: download textures as well
    return _download_and_read('doorman/doorman.obj', load=load)


def download_mug(load=True):  # pragma: no cover
    """Download mug dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.MultiBlock or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_mug()  # doctest:+SKIP

    """
    return _download_and_read('mug.e', load=load)


def download_oblique_cone(load=True):  # pragma: no cover
    """Download oblique cone dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_oblique_cone()  # doctest:+SKIP

    """
    return _download_and_read('ObliqueCone.vtp', load=load)


def download_emoji(load=True):  # pragma: no cover
    """Download emoji dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UniformGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_emoji()  # doctest:+SKIP

    """
    return _download_and_read('emote.jpg', load=load)


def download_emoji_texture(load=True):  # pragma: no cover
    """Download emoji texture.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.Texture or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_emoji_texture()  # doctest:+SKIP

    """
    return _download_and_read('emote.jpg', texture=True, load=load)


def download_teapot(load=True):  # pragma: no cover
    """Download teapot dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_teapot()  # doctest:+SKIP

    This dataset is used in the following examples:

    * :ref:`read_file_example`
    * :ref:`cell_centers_example`

    """
    return _download_and_read('teapot.g', load=load)


def download_brain(load=True):  # pragma: no cover
    """Download brain dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UniformGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_brain()  # doctest:+SKIP

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
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.StructuredGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_structured_grid()  # doctest:+SKIP

    """
    return _download_and_read('StructuredGrid.vts', load=load)


def download_structured_grid_two(load=True):  # pragma: no cover
    """Download structured grid two dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.StructuredGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_structured_grid_two()  # doctest:+SKIP

    """
    return _download_and_read('SampleStructGrid.vtk', load=load)


def download_trumpet(load=True):  # pragma: no cover
    """Download trumpet dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_trumpet()  # doctest:+SKIP

    """
    return _download_and_read('trumpet.obj', load=load)


def download_face(load=True):  # pragma: no cover
    """Download face dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_face()  # doctest:+SKIP

    See :ref:`decimate_example` for an example using
    this dataset.


    """
    # TODO: there is a texture with this
    return _download_and_read('fran_cut.vtk', load=load)


def download_sky_box_nz(load=True):  # pragma: no cover
    """Download skybox-nz dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UniformGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_sky_box_nz()  # doctest:+SKIP

    """
    return _download_and_read('skybox-nz.jpg', load=load)


def download_sky_box_nz_texture(load=True):  # pragma: no cover
    """Download skybox-nz texture.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.Texture or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_sky_box_nz_texture()  # doctest:+SKIP

    """
    return _download_and_read('skybox-nz.jpg', texture=True, load=load)


def download_disc_quads(load=True):  # pragma: no cover
    """Download disc quads dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_disc_quads()  # doctest:+SKIP

    """
    return _download_and_read('Disc_BiQuadraticQuads_0_0.vtu', load=load)


def download_honolulu(load=True):  # pragma: no cover
    """Download honolulu dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_honolulu()  # doctest:+SKIP

    """
    return _download_and_read('honolulu.vtk', load=load)


def download_motor(load=True):  # pragma: no cover
    """Download motor dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_motor()  # doctest:+SKIP

    """
    return _download_and_read('motor.g', load=load)


def download_tri_quadratic_hexahedron(load=True):  # pragma: no cover
    """Download tri quadratic hexahedron dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_tri_quadratic_hexahedron()  # doctest:+SKIP

    """
    return _download_and_read('TriQuadraticHexahedron.vtu', load=load)


def download_human(load=True):  # pragma: no cover
    """Download human dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_human()  # doctest:+SKIP

    """
    return _download_and_read('Human.vtp', load=load)


def download_vtk(load=True):  # pragma: no cover
    """Download vtk dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_vtk()  # doctest:+SKIP

    """
    return _download_and_read('vtk.vtp', load=load)


def download_spider(load=True):  # pragma: no cover
    """Download spider dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_spider()  # doctest:+SKIP

    """
    return _download_and_read('spider.ply', load=load)


def download_carotid(load=True):  # pragma: no cover
    """Download carotid dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UniformGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_carotid()  # doctest:+SKIP

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
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_blow()  # doctest:+SKIP

    """
    return _download_and_read('blow.vtk', load=load)


def download_shark(load=True):  # pragma: no cover
    """Download shark dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_shark()  # doctest:+SKIP

    """
    return _download_and_read('shark.ply', load=load)


def download_dragon(load=True):  # pragma: no cover
    """Download dragon dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_dragon()  # doctest:+SKIP

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
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_armadillo()  # doctest:+SKIP

    """
    return _download_and_read('Armadillo.ply', load=load)


def download_gears(load=True):  # pragma: no cover
    """Download gears dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_gears()  # doctest:+SKIP

    """
    return _download_and_read('gears.stl', load=load)


def download_torso(load=True):  # pragma: no cover
    """Download torso dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_torso()  # doctest:+SKIP

    """
    return _download_and_read('Torso.vtp', load=load)


def download_kitchen(split=False, load=True):  # pragma: no cover
    """Download structured grid of kitchen with velocity field.

    Use the ``split`` argument to extract all of the furniture in the
    kitchen.

    Parameters
    ----------
    split : bool, optional
        Optionally split the furniture and return a
        :class:`pyvista.MultiBlock`.

    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.StructuredGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_kitchen()  # doctest:+SKIP

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
    for key, extent in extents.items():
        alg = _vtk.vtkStructuredGridGeometryFilter()
        alg.SetInputDataObject(mesh)
        alg.SetExtent(extent)
        alg.Update()
        result = pyvista.filters._get_output(alg)
        kitchen[key] = result
    return kitchen


def download_tetra_dc_mesh():  # pragma: no cover
    """Download two meshes defining an electrical inverse problem.

    This contains a high resolution forward modeled mesh and a coarse
    inverse modeled mesh.

    Returns
    -------
    pyvista.MultiBlock
        DataSet.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_tetra_dc_mesh()  # doctest:+SKIP

    """
    local_path, _ = _download_file('dc-inversion.zip')
    filename = os.path.join(local_path, 'mesh-forward.vtu')
    fwd = pyvista.read(filename)
    fwd.set_active_scalars('Resistivity(log10)-fwd')
    filename = os.path.join(local_path, 'mesh-inverse.vtu')
    inv = pyvista.read(filename)
    inv.set_active_scalars('Resistivity(log10)')
    return pyvista.MultiBlock({'forward': fwd, 'inverse': inv})


def download_model_with_variance(load=True):  # pragma: no cover
    """Download model with variance dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_model_with_variance()  # doctest:+SKIP

    See :ref:`plot_opacity_example` for an example using this dataset.

    """
    return _download_and_read("model_with_variance.vtu", load=load)


def download_thermal_probes(load=True):  # pragma: no cover
    """Download thermal probes dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_thermal_probes()  # doctest:+SKIP

    See :ref:`interpolate_example` for an example using this dataset.

    """
    return _download_and_read("probes.vtp", load=load)


def download_carburator(load=True):  # pragma: no cover
    """Download scan of a carburator.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_carburator()  # doctest:+SKIP

    """
    return _download_and_read("carburetor.ply", load=load)


def download_turbine_blade(load=True):  # pragma: no cover
    """Download scan of a turbine blade.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_turbine_blade()  # doctest:+SKIP

    """
    return _download_and_read('turbineblade.ply', load=load)


def download_pine_roots(load=True):  # pragma: no cover
    """Download pine roots dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_pine_roots()  # doctest:+SKIP

    See :ref:`connectivity_example` for an example using this dataset.

    """
    return _download_and_read('pine_root.tri', load=load)


def download_crater_topo(load=True):  # pragma: no cover
    """Download crater dataset.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UniformGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_crater_topo()  # doctest:+SKIP

    This dataset is used in the following examples:

    * :ref:`terrain_following_mesh_example`
    * :ref:`ref_topo_map_example`

    """
    return _download_and_read('Ruapehu_mag_dem_15m_NZTM.vtk', load=load)


def download_crater_imagery(load=True):  # pragma: no cover
    """Download crater texture.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.Texture or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_crater_imagery()  # doctest:+SKIP

    See :ref:`ref_topo_map_example` for an example using this dataset.

    """
    return _download_and_read('BJ34_GeoTifv1-04_crater_clip.tif', texture=True, load=load)


def download_dolfin(load=True):  # pragma: no cover
    """Download dolfin mesh.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_dolfin()  # doctest:+SKIP

    """
    return _download_and_read('dolfin_fine.xml', file_format="dolfin-xml", load=load)


def download_damavand_volcano(load=True):  # pragma: no cover
    """Download damavand volcano model.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UniformGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_damavand_volcano()  # doctest:+SKIP

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
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_delaunay_example()  # doctest:+SKIP

    """
    return _download_and_read('250.vtk', load=load)


def download_embryo(load=True):  # pragma: no cover
    """Download a volume of an embryo.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UniformGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_embryo()  # doctest:+SKIP

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
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_antarctica_velocity()  # doctest:+SKIP

    See :ref:`antarctica_example` for an example using this dataset.

    """
    return _download_and_read("antarctica_velocity.vtp", load=load)


def download_room_surface_mesh(load=True):  # pragma: no cover
    """Download the room surface mesh.

    This mesh is for demonstrating the difference that depth peeling can
    provide whenn rendering translucent geometries.

    This mesh is courtesy of `Sam Potter <https://github.com/sampotter>`_.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_room_surface_mesh()  # doctest:+SKIP

    See :ref:`depth_peeling_example` for an example using this dataset.

    """
    return _download_and_read("room_surface_mesh.obj", load=load)


def download_beach(load=True):  # pragma: no cover
    """Download the beach NRRD image.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UniformGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_beach()  # doctest:+SKIP

    """
    return _download_and_read("beach.nrrd", load=load)


def download_rgba_texture(load=True):  # pragma: no cover
    """Download a texture with an alpha channel.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.Texture or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_rgba_texture()  # doctest:+SKIP

    See :ref:`ref_texture_example` for an example using this dataset.

    """
    return _download_and_read("alphachannel.png", texture=True, load=load)


def download_vtk_logo(load=True):  # pragma: no cover
    """Download a texture of the VTK logo.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.Texture or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_vtk_logo()  # doctest:+SKIP

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
    >>> dataset = examples.download_sky_box_cube_map()  # doctest:+SKIP

    See :ref:`pbr_example` for an example using this dataset.

    """
    prefix = 'skybox2-'
    sets = ['posx', 'negx', 'posy', 'negy', 'posz', 'negz']
    images = [prefix + suffix + '.jpg' for suffix in sets]
    for image in images:
        _download_file(image)

    return pyvista.cubemap(pyvista.EXAMPLES_PATH, prefix)


def download_backward_facing_step(load=True):  # pragma: no cover
    """Download an ensight gold case of a fluid simulation.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.MultiBlock or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_backward_facing_step()  # doctest:+SKIP

    """
    folder, _ = _download_file('EnSight.zip')
    filename = os.path.join(folder, "foam_case_0_0_0_0.case")
    if not load:
        return filename
    return pyvista.read(filename)


def download_gpr_data_array(load=True):  # pragma: no cover
    """Download GPR example data array.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    numpy.ndarray or str
        Array or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_gpr_data_array()  # doctest:+SKIP

    See :ref:`create_draped_surf_example` for an example using this dataset.

    """
    saved_file, _ = _download_file("gpr-example/data.npy")
    if not load:
        return saved_file
    return np.load(saved_file)


def download_gpr_path(load=True):  # pragma: no cover
    """Download GPR example path.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_gpr_path()  # doctest:+SKIP

    See :ref:`create_draped_surf_example` for an example using this dataset.

    """
    saved_file, _ = _download_file("gpr-example/path.txt")
    if not load:
        return saved_file
    path = np.loadtxt(saved_file, skiprows=1)
    return pyvista.PolyData(path)


def download_woman(load=True):  # pragma: no cover
    """Download scan of a woman.

    Originally obtained from Laser Design.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_woman()  # doctest:+SKIP

    """
    return _download_and_read('woman.stl', load=load)


def download_lobster(load=True):  # pragma: no cover
    """Download scan of a lobster.

    Originally obtained from Laser Design.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_lobster()  # doctest:+SKIP

    """
    return _download_and_read('lobster.ply', load=load)


def download_face2(load=True):  # pragma: no cover
    """Download scan of a man's face.

    Originally obtained from Laser Design.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_face2()  # doctest:+SKIP

    """
    return _download_and_read('man_face.stl', load=load)


def download_urn(load=True):  # pragma: no cover
    """Download scan of a burial urn.

    Originally obtained from Laser Design.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_urn()  # doctest:+SKIP

    """
    return _download_and_read('urn.stl', load=load)


def download_pepper(load=True):  # pragma: no cover
    """Download scan of a pepper (capsicum).

    Originally obtained from Laser Design.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_pepper()  # doctest:+SKIP

    """
    return _download_and_read('pepper.ply', load=load)


def download_drill(load=True):  # pragma: no cover
    """Download scan of a power drill.

    Originally obtained from Laser Design.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_drill()  # doctest:+SKIP

    """
    return _download_and_read('drill.obj', load=load)


def download_action_figure(load=True):  # pragma: no cover
    """Download scan of an action figure.

    Originally obtained from Laser Design.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_action_figure()  # doctest:+SKIP

    """
    return _download_and_read('tigerfighter.obj', load=load)


def download_mars_jpg():  # pragma: no cover
    """Download and return the path of ``'mars.jpg'``.

    Returns
    -------
    str
        Filename of the JPEG.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_mars_jpg()  # doctest:+SKIP

    """
    return _download_file('mars.jpg')[0]


def download_stars_jpg():  # pragma: no cover
    """Download and return the path of ``'stars.jpg'``.

    Returns
    -------
    str
        Filename of the JPEG.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_stars_jpg()  # doctest:+SKIP

    """
    return _download_file('stars.jpg')[0]


def download_notch_stress(load=True):  # pragma: no cover
    """Download the FEA stress result from a notched beam.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid or str
        DataSet or filename depending on ``load``.

    Notes
    -----
    This file may have issues being read in on VTK 8.1.2

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_notch_stress()  # doctest:+SKIP

    """
    return _download_and_read('notch_stress.vtk', load=load)


def download_notch_displacement(load=True):  # pragma: no cover
    """Download the FEA displacement result from a notched beam.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.UnstructuredGrid or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_notch_displacement()  # doctest:+SKIP

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
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_louis_louvre()  # doctest:+SKIP

    See :ref:`pbr_example` for an example using this dataset.

    """
    return _download_and_read('louis.ply', load=load)


def download_cylinder_crossflow(load=True):  # pragma: no cover
    """Download CFD result for cylinder in cross flow at Re=35.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.MultiBlock or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_cylinder_crossflow()  # doctest:+SKIP

    See :ref:`2d_streamlines_example` for an example using this dataset.

    """
    filename, _ = _download_file('EnSight/CylinderCrossflow/cylinder_Re35.case')
    _download_file('EnSight/CylinderCrossflow/cylinder_Re35.geo')
    _download_file('EnSight/CylinderCrossflow/cylinder_Re35.scl1')
    _download_file('EnSight/CylinderCrossflow/cylinder_Re35.scl2')
    _download_file('EnSight/CylinderCrossflow/cylinder_Re35.vel')
    if not load:
        return filename
    return pyvista.read(filename)


def download_naca(load=True):  # pragma: no cover
    """Download NACA airfoil dataset in EnSight format.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.MultiBlock or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_naca()  # doctest:+SKIP

    See :ref:`reader_example` for an example using this dataset.

    """
    filename, _ = _download_file('EnSight/naca.bin.case')
    _download_file('EnSight/naca.gold.bin.DENS_1')
    _download_file('EnSight/naca.gold.bin.DENS_3')
    _download_file('EnSight/naca.gold.bin.geo')
    if not load:
        return filename
    return pyvista.read(filename)


def download_wavy(load=True):  # pragma: no cover
    """Download PVD file of a 2D wave.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.MultiBlock or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_wavy()  # doctest:+SKIP

    See :ref:`reader_example` for an example using this dataset.

    """
    folder, _ = _download_file('PVD/wavy.zip')
    filename = os.path.join(folder, 'wavy.pvd')
    if not load:
        return filename
    return pyvista.PVDReader(filename).read()


def download_single_sphere_animation(load=True):  # pragma: no cover
    """Download PVD file for single sphere.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.MultiBlock or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_single_sphere_animation()  # doctest:+SKIP

    """
    filename, _ = _download_file('PVD/paraview/singleSphereAnimation.pvd')
    folder, _ =_download_file('PVD/paraview/singleSphereAnimation')
    if not load:
        return filename
    return pyvista.PVDReader(filename).read()


def download_dual_sphere_animation(load=True):  # pragma: no cover
    """Download PVD file for double sphere.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.MultiBlock or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_dual_sphere_animation()  # doctest:+SKIP

    """
    filename, _ = _download_file('PVD/paraview/dualSphereAnimation.pvd')
    folder, _ =_download_file('PVD/paraview/dualSphereAnimation')
    if not load:
        return filename
    return pyvista.PVDReader(filename).read()


def download_osmnx_graph():  # pragma: no cover
    """Load a simple street map from Open Street Map.

    Generated from:

    .. code:: python

        >>> import osmnx as ox  # doctest:+SKIP
        >>> address = 'Holzgerlingen DE'  # doctest:+SKIP
        >>> graph = ox.graph_from_address(address, dist=500, network_type='drive')  # doctest:+SKIP
        >>> pickle.dump(graph, open('osmnx_graph.p', 'wb'))  # doctest:+SKIP

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
        import osmnx
    except ImportError:
        raise ImportError('Install `osmnx` to use this example')

    filename, _ = _download_file('osmnx_graph.p')
    return pickle.load(open(filename, 'rb'))


def download_lucy(load=True):  # pragma: no cover
    """Download the lucy angel mesh.

    Original downloaded from the `The Stanford 3D Scanning Repository
    <http://graphics.stanford.edu/data/3Dscanrep/>`_ and decimated to
    approximately 100k triangle.

    Parameters
    ----------
    load : bool, optional
        Load the dataset after downloading it when ``True``.  Set this
        to ``False`` and only the filename will be returned.

    Returns
    -------
    pyvista.PolyData or str
        DataSet or filename depending on ``load``.

    Examples
    --------
    >>> from pyvista import examples
    >>> dataset = examples.download_lucy()  # doctest:+SKIP

    See :ref:`jupyter_plotting` for an example using this dataset.

    """
    return _download_and_read('lucy.ply', load=load)
