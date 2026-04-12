"""These are private methods we keep out of plotting.py to simplify the module."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import numpy as np

from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista._warn_external import warn_external
from pyvista.core.utilities.arrays import get_array
from pyvista.core.utilities.misc import assert_empty_kwargs

from .colors import Color
from .opts import InterpolationType
from .tools import opacity_transfer_function

if TYPE_CHECKING:
    from pyvista.core._typing_core import NumpyArray
    from pyvista.core.dataset import DataSet
    from pyvista.core.utilities.arrays import CellLiteral
    from pyvista.core.utilities.arrays import PointLiteral


def _resolve_scalars_field(
    scalars: NumpyArray[float],
    mesh: DataSet,
    preference: PointLiteral | CellLiteral,
) -> PointLiteral | CellLiteral:
    """Decide whether raw numpy ``scalars`` attach to points or cells.

    Matches by length. When both dimensions coincide (``n_points ==
    n_cells``), falls back to ``preference``. Raises ``ValueError`` when
    the array length matches neither — callers previously would silently
    drop the array and fail cryptically downstream.
    """
    matches_points = scalars.shape[0] == mesh.n_points
    matches_cells = scalars.shape[0] == mesh.n_cells
    if matches_points and matches_cells:
        return preference
    if matches_points:
        return 'point'
    if matches_cells:
        return 'cell'
    msg = (
        f'Length of scalars array ({scalars.shape[0]}) must match either the '
        f'number of points ({mesh.n_points}) or cells ({mesh.n_cells}) in the mesh.'
    )
    raise ValueError(msg)


def reduce_component_scalars(
    scalars: NumpyArray[float],
    scalars_name: str,
    component: int | None,
) -> tuple[NumpyArray[float], str]:
    """Reduce a 2D scalar array to 1D by magnitude or component index.

    Produces the derived array and synthesized name (``{name}-normed`` for
    magnitude or ``{name}-{component}`` for a component pick) that
    :meth:`DataSetMapper.set_scalars` and smooth-shading pre-processing
    both rely on. Keeping this in one place prevents the two call sites
    from drifting on naming or bounds checks.

    Parameters
    ----------
    scalars : numpy.ndarray
        2D scalar array of shape ``(n, k)``.

    scalars_name : str
        Base name of the input array.

    component : int | None
        Component index to extract, or ``None`` for vector magnitude.

    Returns
    -------
    reduced : numpy.ndarray
        1D array of shape ``(n,)``.

    derived_name : str
        Synthesized name carrying the reduction semantics.

    """
    if component is None:
        return np.linalg.norm(scalars, axis=1), f'{scalars_name}-normed'
    try:
        component_index = operator.index(component)
    except TypeError:
        msg = 'component must be None or an integer.'
        raise TypeError(msg) from None
    if not 0 <= component_index < scalars.shape[1]:
        msg = (
            'component must be nonnegative and less than the '
            f'dimensionality of the scalars array: {scalars.shape[1]}'
        )
        raise ValueError(msg)
    return np.array(scalars[:, component_index]), f'{scalars_name}-{component_index}'


@_deprecate_positional_args
def process_opacity(mesh, opacity, preference, n_colors, scalars, use_transparency):  # noqa: PLR0917
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
                warn_external('Opacity scalars contain values over 1')  # pragma: no cover
            if np.any(opacity < 0):
                warn_external('Opacity scalars contain values less than 0')  # pragma: no cover
            custom_opac = True
        except KeyError:
            # Or get opacity transfer function (e.g. "linear")
            opacity = opacity_transfer_function(opacity, n_colors)
        else:
            if scalars.shape[0] != opacity.shape[0]:
                msg = 'Opacity array and scalars array must have the same number of elements.'
                raise ValueError(msg)
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
    *,
    dataset,
    theme,
    n_colors,
    scalar_bar_args,
    split_sharp_edges,
    show_scalar_bar,
    render_points_as_spheres,
    point_shape,
    smooth_shading,
    pbr,
    clim,
    cmap,
    culling,
    name,
    nan_color,
    nan_opacity,
    texture,
    rgb,
    style,
    remove_existing_actor=None,
    **kwargs,
):
    """Parse arguments in common between add_volume, composite, and mesh."""
    # supported aliases
    clim = kwargs.pop('rng', clim)
    cmap = kwargs.pop('colormap', cmap)
    culling = kwargs.pop('backface_culling', culling)
    rgb = kwargs.pop('rgba', rgb)
    vertex_color = kwargs.pop('vertex_color', theme.edge_color)
    vertex_style = kwargs.pop('vertex_style', 'points')
    vertex_opacity = kwargs.pop('vertex_opacity', 1.0)

    # Support aliases for 'back', 'front', or 'none'. Consider deprecating
    if culling is False:
        culling = 'none'
    elif culling in ['b', 'backface', True]:
        culling = 'back'
    elif culling in ['f', 'frontface']:
        culling = 'front'

    if show_scalar_bar is None:
        # use theme unless plotting RGB
        _default = theme.show_scalar_bar or scalar_bar_args
        show_scalar_bar = False if rgb else _default
    # Avoid mutating input
    scalar_bar_args = {'n_colors': n_colors} if scalar_bar_args is None else scalar_bar_args.copy()

    # theme based parameters
    if split_sharp_edges is None:
        split_sharp_edges = theme.split_sharp_edges
    feature_angle = kwargs.pop('feature_angle', theme.sharp_edges_feature_angle)
    if render_points_as_spheres is None:
        if style == 'points_gaussian':
            render_points_as_spheres = False
        else:
            render_points_as_spheres = theme.render_points_as_spheres

    if point_shape is None:
        point_shape = theme.point_shape

    if point_shape is not None and render_points_as_spheres:
        warn_external(
            f'point_shape={point_shape!r} requires render_points_as_spheres=False. '
            'Disabling render_points_as_spheres.',
        )
        render_points_as_spheres = False

    if smooth_shading is None:
        smooth_shading = True if pbr else theme.smooth_shading

    if name is None:
        name = f'{type(dataset).__name__}({dataset.memory_address})'
        # Default to False when no name is provided
        if remove_existing_actor is None:  # pragma: no cover
            remove_existing_actor = False
    # Default to True when a name is provided (for backwards compatibility)
    elif remove_existing_actor is None:
        remove_existing_actor = True

    nan_color = Color(nan_color, opacity=nan_opacity, default_color=theme.nan_color)

    if texture is False:
        texture = None

    # allow directly specifying interpolation (potential future feature)
    if 'interpolation' in kwargs:
        interpolation = kwargs.pop('interpolation')  # pragma: no cover:
    elif pbr:
        interpolation = InterpolationType.PBR
    elif smooth_shading:
        interpolation = InterpolationType.PHONG
    else:
        interpolation = theme.lighting_params.interpolation

    if 'scalar' in kwargs:
        msg = '`scalar` is an invalid keyword argument. Perhaps you mean `scalars` with an s?'
        raise TypeError(msg)

    assert_empty_kwargs(**kwargs)
    return (
        scalar_bar_args,
        split_sharp_edges,
        show_scalar_bar,
        feature_angle,
        render_points_as_spheres,
        point_shape,
        smooth_shading,
        clim,
        cmap,
        culling,
        name,
        nan_color,
        texture,
        rgb,
        interpolation,
        remove_existing_actor,
        vertex_color,
        vertex_style,
        vertex_opacity,
    )
