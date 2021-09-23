"""These are private methods we keep out of plotting.py to simplfy the module."""
import warnings

import numpy as np
import pyvista

from pyvista.utilities import get_array, is_pyvista_dataset
from .tools import opacity_transfer_function


def _has_matplotlib():
    try:
        import matplotlib
        return True
    except ImportError:  # pragma: no cover
        return False


def prepare_smooth_shading(mesh, scalars, texture, split_sharp_edges, feature_angle):
    """Prepare a dataset for smooth shading.

    VTK requires datasets with prong shading to have active normals.
    This requires extracting the external surfaces from non-polydata
    datasets and computing the point normals.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Dataset to prepare smooth shading for.
    texture : vtk.vtkTexture or np.ndarray or bool, optional
        A texture to apply to the mesh.
    split_sharp_edges : bool, optional
        Split sharp edges exceeding 30 degrees when plotting with
        smooth shading.  Control the angle with the optional
        keyword argument ``feature_angle``.  By default this is
        ``False``.  Note that enabling this will create a copy of
        the input mesh within the plotter.  See
        :ref:`shading_example`.
    feature_angle : float, optional
        Angle to consider an edge a sharp edge.

    Returns
    -------
    pyvista.PolyData
        Always a surface as we need to compute point normals.

    """
    # extract surface if mesh is exterior
    if not isinstance(mesh, pyvista.PolyData):
        grid = mesh
        mesh = grid.extract_surface()
        # remap scalars
        if scalars is not None:
            ind = mesh.point_data['vtkOriginalPointIds']
            scalars = np.asarray(scalars[ind])
    if texture:
        _tcoords = mesh.active_t_coords

    if split_sharp_edges:
        # we must track the original IDs
        indices = np.arange(mesh.n_points, dtype=np.int32)
        mesh.point_data['__orig_ids__'] = indices
        mesh = mesh.compute_normals(
            cell_normals=False,
            split_vertices=True,
            feature_angle=feature_angle,
        )
        if scalars is not None:
            ind = mesh.point_data['__orig_ids__']
            scalars = np.asarray(scalars)[ind]
    else:
        mesh.compute_normals(cell_normals=False, inplace=True)

    if texture:
        mesh.active_t_coords = _tcoords

    return mesh, scalars


def process_opacity(mesh, opacity, preference, n_colors, scalars, use_transparency):
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
        except:
            # Or get opacity transfer function
            opacity = opacity_transfer_function(opacity, n_colors)
        else:
            if scalars.shape[0] != opacity.shape[0]:
                raise ValueError(
                    "Opacity array and scalars array must have the same number "
                    "of elements."
                )
    elif isinstance(opacity, (np.ndarray, list, tuple)):
        opacity = np.array(opacity)
        if scalars.shape[0] == opacity.shape[0]:
            # User could pass an array of opacities for every point/cell
            _custom_opac = True
        else:
            opacity = opacity_transfer_function(opacity, n_colors)

    if use_transparency and np.max(opacity) <= 1.0:
        opacity = 1 - opacity
    elif use_transparency and isinstance(opacity, np.ndarray):
        opacity = 255 - opacity

    return _custom_opac, opacity
