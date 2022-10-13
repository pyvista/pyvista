"""These are private methods we keep out of plotting.py to simplify the module."""
import warnings

import numpy as np

import pyvista
from pyvista.utilities import assert_empty_kwargs, get_array

from ..utilities.misc import PyVistaDeprecationWarning
from .colors import Color
from .tools import opacity_transfer_function

USE_SCALAR_BAR_ARGS = """
"stitle" is a deprecated keyword argument and will be removed in a future
release.

Use ``scalar_bar_args`` instead.  For example:

scalar_bar_args={'title': 'Scalar Bar Title'}
"""


def prepare_smooth_shading(mesh, scalars, texture, split_sharp_edges, feature_angle, preference):
    """Prepare a dataset for smooth shading.

    VTK requires datasets with Phong shading to have active normals.
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
        Either ``'point'`` or ``'cell'``.

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
        if scalars.shape[0] == mesh.n_points and scalars.shape[0] == mesh.n_cells:
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
        mesh = mesh.compute_normals(
            cell_normals=False,
            split_vertices=True,
            feature_angle=feature_angle,
        )
        if is_polydata:
            if has_scalars and use_points:
                # we must track the original IDs with our own array from compute_normals
                indices_array = 'pyvistaOriginalPointIds'
    else:
        # consider checking if mesh contains active normals
        # if mesh.point_data.active_normals is None:
        mesh.compute_normals(cell_normals=False, inplace=True)

    if has_scalars and indices_array is not None:
        ind = mesh[indices_array]
        scalars = np.asarray(scalars)[ind]

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

    preference : str
        When ``mesh.n_points == mesh.n_cells``, this parameter
        sets how the scalars will be mapped to the mesh. If
        ``'point'``, causes the scalars will be associated with
        the mesh points.  Can be either ``'point'`` or
        ``'cell'``.

    n_colors : int
        Number of colors to use when displaying the opacity.

    scalars : numpy.ndarray
        Dataset scalars.

    use_transparency : bool
        Invert the opacity mappings and make the values correspond
        to transparency.

    Returns
    -------
    custom_opac : bool
        If using custom opacity.

    opacity : numpy.ndarray
        Array containing the opacity.

    """
    custom_opac = False
    if isinstance(opacity, str):
        try:
            # Get array from mesh
            opacity = get_array(mesh, opacity, preference=preference, err=True)
            if np.any(opacity > 1):
                warnings.warn("Opacity scalars contain values over 1")
            if np.any(opacity < 0):
                warnings.warn("Opacity scalars contain values less than 0")
            custom_opac = True
        except KeyError:
            # Or get opacity transfer function (e.g. "linear")
            opacity = opacity_transfer_function(opacity, n_colors)
        else:
            if scalars.shape[0] != opacity.shape[0]:
                raise ValueError(
                    "Opacity array and scalars array must have the same number of elements."
                )
    elif isinstance(opacity, (np.ndarray, list, tuple)):
        opacity = np.asanyarray(opacity)
        if opacity.shape[0] in [mesh.n_cells, mesh.n_points]:
            # User could pass an array of opacities for every point/cell
            custom_opac = True
        else:
            opacity = opacity_transfer_function(opacity, n_colors)

    if use_transparency:
        if np.max(opacity) <= 1.0:
            opacity = 1 - opacity
        elif isinstance(opacity, np.ndarray):
            opacity = 255 - opacity

    return custom_opac, opacity


def _common_arg_parser(
    dataset,
    theme,
    n_colors,
    scalar_bar_args,
    split_sharp_edges,
    show_scalar_bar,
    render_points_as_spheres,
    smooth_shading,
    pbr,
    clim,
    cmap,
    culling,
    name,
    nan_color,
    nan_opacity,
    color,
    texture,
    rgb,
    style,
    **kwargs,
):
    """Parse arguments in common between add_volume, composite, and mesh."""
    # supported aliases
    clim = kwargs.pop('rng', clim)
    cmap = kwargs.pop('colormap', cmap)
    culling = kwargs.pop("backface_culling", culling)
    rgb = kwargs.pop('rgba', rgb)

    # Support aliases for 'back', 'front', or 'none'. Consider deprecating
    if culling is False:
        culling = 'none'
    elif culling in ['b', 'backface', True]:
        culling = 'back'
    elif culling in ['f', 'frontface']:
        culling = 'front'

    # Avoid mutating input
    if scalar_bar_args is None:
        scalar_bar_args = {'n_colors': n_colors}
    else:
        scalar_bar_args = scalar_bar_args.copy()

    # theme based parameters
    if split_sharp_edges is None:
        split_sharp_edges = theme.split_sharp_edges
    if show_scalar_bar is None:
        # use theme unless plotting RGB
        show_scalar_bar = False if rgb else theme.show_scalar_bar
    feature_angle = kwargs.pop('feature_angle', theme.sharp_edges_feature_angle)
    if render_points_as_spheres is None:
        if style == 'points_gaussian':
            render_points_as_spheres = True
        else:
            render_points_as_spheres = theme.render_points_as_spheres

    if smooth_shading is None:
        if pbr:
            smooth_shading = True
        else:
            smooth_shading = theme.smooth_shading

    if name is None:
        name = f'{type(dataset).__name__}({dataset.memory_address})'
        remove_existing_actor = False
    else:
        # check if this actor already exists
        remove_existing_actor = True

    nan_color = Color(nan_color, default_opacity=nan_opacity, default_color=theme.nan_color)

    if color is True:
        color = theme.color

    if texture is False:
        texture = None

    # allow directly specifying interpolation (potential future feature)
    if 'interpolation' in kwargs:
        interpolation = kwargs.pop('interpolation')  # pragma: no cover:
    else:
        if pbr:
            interpolation = 'Physically based rendering'
        elif smooth_shading:
            interpolation = 'Phong'
        else:
            interpolation = 'Flat'

    # account for legacy behavior
    if 'stitle' in kwargs:  # pragma: no cover
        warnings.warn(USE_SCALAR_BAR_ARGS, PyVistaDeprecationWarning)
        scalar_bar_args.setdefault('title', kwargs.pop('stitle'))

    if "scalar" in kwargs:
        raise TypeError(
            "`scalar` is an invalid keyword argument. Perhaps you mean `scalars` with an s?"
        )

    assert_empty_kwargs(**kwargs)
    return (
        scalar_bar_args,
        split_sharp_edges,
        show_scalar_bar,
        feature_angle,
        render_points_as_spheres,
        smooth_shading,
        clim,
        cmap,
        culling,
        name,
        nan_color,
        color,
        texture,
        rgb,
        interpolation,
        remove_existing_actor,
    )
