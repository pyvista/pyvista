"""These are private methods we keep out of plotting.py to simplfy the module."""
import warnings

import numpy as np

import pyvista
from pyvista.utilities import get_array

from .tools import opacity_transfer_function


def _has_matplotlib():
    try:
        import matplotlib  # noqa
        return True
    except ImportError:  # pragma: no cover
        return False


def prepare_smooth_shading(mesh, scalars, texture, split_sharp_edges, feature_angle,
                           preference):
    """Prepare a dataset for smooth shading.

    VTK requires datasets with prong shading to have active normals.
    This requires extracting the external surfaces from non-polydata
    datasets and computing the point normals.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Dataset to prepare smooth shading for.
    scalars : sequence
        Sequence of scalars.
    texture : vtk.vtkTexture or np.ndarray or bool, optional
        A texture to apply to the mesh.
    split_sharp_edges : bool
        Split sharp edges exceeding 30 degrees when plotting with
        smooth shading.  Control the angle with the optional
        keyword argument ``feature_angle``.  By default this is
        ``False``.  Note that enabling this will create a copy of
        the input mesh within the plotter.  See
        :ref:`shading_example`.
    feature_angle : float
        Angle to consider an edge a sharp edge.
    preference : str
        If the number of points is identical to the number of cells.
        Either ``'point'`` or '``cell'``.

    Returns
    -------
    pyvista.PolyData
        Always a surface as we need to compute point normals.

    """
    is_polydata = isinstance(mesh, pyvista.PolyData)
    indices_array = None

    has_scalars = scalars is not None
    use_points = False
    if has_scalars:
        if not isinstance(scalars, np.ndarray):
            scalars = np.array(scalars)
        if (scalars.shape[0] == mesh.n_points and scalars.shape[0] == mesh.n_cells):
            use_points = preference == 'point'
        else:
            use_points = scalars.shape[0] == mesh.n_points

    # extract surface if not already a surface
    if not is_polydata:
        mesh = mesh.extract_surface(
            pass_pointid=use_points or texture is not None,
            pass_cellid=not use_points,
        )
        if use_points:
            indices_array = 'vtkOriginalPointIds'
        else:
            indices_array = 'vtkOriginalCellIds'

    if split_sharp_edges:
        if is_polydata:
            if has_scalars:
                # we must track the original IDs with our own array
                indices_array = '__orig_ids__'
                if use_points:
                    arr_sz = mesh.n_points
                else:
                    arr_sz = mesh.n_cells
                mesh.point_data[indices_array] = np.arange(arr_sz, dtype=np.int32)
        mesh = mesh.compute_normals(
            cell_normals=False,
            split_vertices=True,
            feature_angle=feature_angle,
        )
    else:
        # consider checking if mesh contains active normals
        # if mesh.point_data.active_normals is None:
        mesh.compute_normals(cell_normals=False, inplace=True)

    if has_scalars and indices_array is not None:
        ind = mesh[indices_array]
        scalars = np.asarray(scalars)[ind]

    # remove temporary indices array
    if indices_array == '__orig_ids__':
        del mesh.point_data['__orig_ids__']

    return mesh, scalars


def process_opacity(mesh, opacity, preference, n_colors, scalars, use_transparency):
    """Process opacity.

    This function accepts an opacity string or array and always
    returns an array that can be applied to a dataset for plotting.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Dataset to process the opacity for.
    opacity : str, sequence
        String or array.  If string, can be a ``str`` name of a
        predefined mapping such as ``'linear'``, ``'geom'``,
        ``'sigmoid'``, ``'sigmoid3-10'``, or the key of a cell or
        point data array.
    preference : str, optional
        When ``mesh.n_points == mesh.n_cells``, this parameter
        sets how the scalars will be mapped to the mesh.  Default
        ``'points'``, causes the scalars will be associated with
        the mesh points.  Can be either ``'points'`` or
        ``'cells'``.
    n_colors : int, optional
        Number of colors to use when displaying the opacity.
    scalars : numpy.ndarray, optional
        Dataset scalars.
    use_transparency : bool, optional
        Invert the opacity mappings and make the values correspond
        to transparency.

    Returns
    -------
    _custom_opac : bool
        If using custom opacity.
    opacity : numpy.ndarray
        Array containing the opacity.

    """
    _custom_opac = False
    if isinstance(opacity, str):
        try:
            # Get array from mesh
            opacity = get_array(mesh, opacity,
                                preference=preference, err=True)
            if np.any(opacity > 1):
                warnings.warn("Opacity scalars contain values over 1")
            if np.any(opacity < 0):
                warnings.warn("Opacity scalars contain values less than 0")
            _custom_opac = True
        except KeyError:
            # Or get opacity transfer function (e.g. "linear")
            opacity = opacity_transfer_function(opacity, n_colors)
        else:
            if scalars.shape[0] != opacity.shape[0]:
                raise ValueError(
                    "Opacity array and scalars array must have the same number "
                    "of elements."
                )
    elif isinstance(opacity, (np.ndarray, list, tuple)):
        opacity = np.array(opacity)
        if opacity.shape[0] in [mesh.n_cells, mesh.n_points]:
            # User could pass an array of opacities for every point/cell
            _custom_opac = True
        else:
            opacity = opacity_transfer_function(opacity, n_colors)

    if use_transparency:
        if np.max(opacity) <= 1.0:
            opacity = 1 - opacity
        elif isinstance(opacity, np.ndarray):
            opacity = 255 - opacity

    return _custom_opac, opacity
