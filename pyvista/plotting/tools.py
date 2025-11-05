"""Module containing useful plotting tools."""

from __future__ import annotations

from enum import Enum
import os
import platform
from subprocess import PIPE
from subprocess import Popen
from subprocess import TimeoutExpired

import numpy as np

import pyvista
from pyvista._deprecate_positional_args import _deprecate_positional_args

from . import _vtk
from .colors import Color


class FONTS(Enum):
    """Font families available to PyVista."""

    arial = _vtk.VTK_ARIAL
    courier = _vtk.VTK_COURIER
    times = _vtk.VTK_TIMES


# Track render window support and plotting
SUPPORTS_OPENGL = None
SUPPORTS_PLOTTING = None


def supports_open_gl():
    """Return if the system supports OpenGL.

    This function checks if the system supports OpenGL by creating a VTK render
    window and querying its OpenGL support.

    Returns
    -------
    bool
        ``True`` if the system supports OpenGL, ``False`` otherwise.

    """
    global SUPPORTS_OPENGL  # noqa: PLW0603
    if SUPPORTS_OPENGL is None:
        ren_win = _vtk.vtkRenderWindow()
        SUPPORTS_OPENGL = bool(ren_win.SupportsOpenGL())
    return SUPPORTS_OPENGL


def _system_supports_plotting() -> bool:  # noqa: PLR0911
    """Check if the environment supports plotting on Windows, Linux, or Mac OS.

    Returns
    -------
    system_supports_plotting : bool
        ``True`` when system supports plotting.

    """
    if os.environ.get('ALLOW_PLOTTING', '').lower() == 'true':
        return True

    # Windows case
    if os.name == 'nt':
        # actually have to check here.  Somewhat expensive.
        return supports_open_gl()

    # mac case
    if platform.system() == 'Darwin':
        # check if finder available
        proc = Popen(['pgrep', '-qx', 'Finder'], stdout=PIPE, stderr=PIPE, encoding='utf8')
        try:
            proc.communicate(timeout=10)
        except TimeoutExpired:
            return False
        if proc.returncode == 0:
            return True

        # display variable set, likely available
        return 'DISPLAY' in os.environ

    # Linux case
    if os.environ.get('WAYLAND_DISPLAY'):  # pragma: no cover
        return True

    try:
        proc = Popen(['xset', '-q'], stdout=PIPE, stderr=PIPE, encoding='utf8')
        proc.communicate(timeout=10)
    except (OSError, TimeoutExpired):  # pragma: no cover
        # possible we have EGL support
        return supports_open_gl()
    else:  # pragma: no cover
        return proc.returncode == 0


def system_supports_plotting() -> bool:
    """Check if the environment supports plotting.

    Returns
    -------
    bool
        ``True`` when system supports plotting.

    """
    global SUPPORTS_PLOTTING  # noqa: PLW0603
    if SUPPORTS_PLOTTING is None:
        SUPPORTS_PLOTTING = _system_supports_plotting()

    # always use the cached response
    return SUPPORTS_PLOTTING


def _update_axes_label_color(axes_actor, color=None):
    """Set the axes label color (internal helper)."""
    color = Color(color, default_color=pyvista.global_theme.font.color)
    if isinstance(axes_actor, _vtk.vtkAxesActor):
        prop_x = axes_actor.GetXAxisCaptionActor2D().GetCaptionTextProperty()
        prop_y = axes_actor.GetYAxisCaptionActor2D().GetCaptionTextProperty()
        prop_z = axes_actor.GetZAxisCaptionActor2D().GetCaptionTextProperty()
        for prop in [prop_x, prop_y, prop_z]:
            prop.SetColor(color.float_rgb)
            prop.SetShadow(False)
    elif isinstance(axes_actor, _vtk.vtkAnnotatedCubeActor):
        axes_actor.GetTextEdgesProperty().SetColor(color.float_rgb)


@_deprecate_positional_args
def create_axes_marker(  # noqa: PLR0917
    label_color=None,
    x_color=None,
    y_color=None,
    z_color=None,
    xlabel='X',
    ylabel='Y',
    zlabel='Z',
    labels_off: bool = False,  # noqa: FBT001, FBT002
    line_width=2,
    cone_radius=0.4,
    shaft_length=0.8,
    tip_length=0.2,
    ambient=0.5,
    label_size=(0.25, 0.1),
) -> _vtk.vtkAxesActor:
    """Create an axis actor.

    Parameters
    ----------
    label_color : ColorLike, optional
        Color of the label text.

    x_color : ColorLike, optional
        Color of the x-axis text.

    y_color : ColorLike, optional
        Color of the y-axis text.

    z_color : ColorLike, optional
        Color of the z-axis text.

    xlabel : str, default: "X"
        Text used for the x-axis.

    ylabel : str, default: "Y"
        Text used for the y-axis.

    zlabel : str, default: "Z"
        Text used for the z-axis.

    labels_off : bool, default: False
        Enable or disable the text labels for the axes.

    line_width : float, default: 2
        The width of the marker lines.

    cone_radius : float, default: 0.4
        The radius of the axes arrow tips.

    shaft_length : float, default: 0.8
        The length of the axes arrow shafts.

    tip_length : float, default: 0.2
        Length of the tip.

    ambient : float, default: 0.5
        The ambient of the axes arrows. Value should be between 0 and 1.

    label_size : sequence[float], default: (0.25, 0.1)
        The width and height of the axes label actors. Values should be between
        0 and 1. For example ``(0.2, 0.1)``.

    Returns
    -------
    :vtk:`vtkAxesActor`
        Axes actor.

    Examples
    --------
    Create the default axes marker.

    >>> import pyvista as pv
    >>> marker = pv.create_axes_marker()
    >>> pl = pv.Plotter()
    >>> _ = pl.add_actor(marker)
    >>> pl.show()

    Create an axes marker at the origin with custom colors and axis labels.

    >>> import pyvista as pv
    >>> marker = pv.create_axes_marker(
    ...     line_width=4,
    ...     ambient=0.0,
    ...     x_color='#378df0',
    ...     y_color='#ab2e5d',
    ...     z_color='#f7fb9a',
    ...     xlabel='X Axis',
    ...     ylabel='Y Axis',
    ...     zlabel='Z Axis',
    ...     label_size=(0.1, 0.1),
    ... )
    >>> pl = pv.Plotter()
    >>> _ = pl.add_actor(marker)
    >>> pl.show()

    """
    x_color = Color(x_color, default_color=pyvista.global_theme.axes.x_color)
    y_color = Color(y_color, default_color=pyvista.global_theme.axes.y_color)
    z_color = Color(z_color, default_color=pyvista.global_theme.axes.z_color)
    axes_actor = _vtk.vtkAxesActor()
    axes_actor.GetXAxisShaftProperty().SetColor(x_color.float_rgb)
    axes_actor.GetXAxisTipProperty().SetColor(x_color.float_rgb)
    axes_actor.GetYAxisShaftProperty().SetColor(y_color.float_rgb)
    axes_actor.GetYAxisTipProperty().SetColor(y_color.float_rgb)
    axes_actor.GetZAxisShaftProperty().SetColor(z_color.float_rgb)
    axes_actor.GetZAxisTipProperty().SetColor(z_color.float_rgb)
    # Set labels
    axes_actor.SetXAxisLabelText(xlabel)
    axes_actor.SetYAxisLabelText(ylabel)
    axes_actor.SetZAxisLabelText(zlabel)
    if labels_off:
        axes_actor.AxisLabelsOff()
    # Set Line width
    axes_actor.GetXAxisShaftProperty().SetLineWidth(line_width)
    axes_actor.GetYAxisShaftProperty().SetLineWidth(line_width)
    axes_actor.GetZAxisShaftProperty().SetLineWidth(line_width)

    axes_actor.SetConeRadius(cone_radius)
    axes_actor.SetNormalizedShaftLength([shaft_length] * 3)
    axes_actor.SetNormalizedTipLength([tip_length] * 3)
    axes_actor.GetXAxisShaftProperty().SetAmbient(ambient)
    axes_actor.GetYAxisShaftProperty().SetAmbient(ambient)
    axes_actor.GetZAxisShaftProperty().SetAmbient(ambient)
    axes_actor.GetXAxisTipProperty().SetAmbient(ambient)
    axes_actor.GetYAxisTipProperty().SetAmbient(ambient)
    axes_actor.GetZAxisTipProperty().SetAmbient(ambient)

    for label_actor in [
        axes_actor.GetXAxisCaptionActor2D(),
        axes_actor.GetYAxisCaptionActor2D(),
        axes_actor.GetZAxisCaptionActor2D(),
    ]:
        label_actor.SetWidth(label_size[0])
        label_actor.SetHeight(label_size[1])

    _update_axes_label_color(axes_actor, label_color)

    return axes_actor


@_deprecate_positional_args
def create_axes_orientation_box(  # noqa: PLR0917
    line_width=1,
    text_scale=0.366667,
    edge_color='black',
    x_color=None,
    y_color=None,
    z_color=None,
    xlabel='X',
    ylabel='Y',
    zlabel='Z',
    x_face_color='red',
    y_face_color='green',
    z_face_color='blue',
    color_box: bool = False,  # noqa: FBT001, FBT002
    label_color=None,
    labels_off: bool = False,  # noqa: FBT001, FBT002
    opacity=0.5,
    show_text_edges: bool = False,  # noqa: FBT001, FBT002
):
    """Create a Box axes orientation widget with labels.

    Parameters
    ----------
    line_width : float, optional
        The width of the marker lines.

    text_scale : float, optional
        Size of the text relative to the faces.

    edge_color : ColorLike, optional
        Color of the edges.

    x_color : ColorLike, optional
        Color of the x-axis text.

    y_color : ColorLike, optional
        Color of the y-axis text.

    z_color : ColorLike, optional
        Color of the z-axis text.

    xlabel : str, optional
        Text used for the x-axis.

    ylabel : str, optional
        Text used for the y-axis.

    zlabel : str, optional
        Text used for the z-axis.

    x_face_color : ColorLike, optional
        Color used for the x-axis arrow.  Defaults to theme axes
        parameters.

    y_face_color : ColorLike, optional
        Color used for the y-axis arrow.  Defaults to theme axes
        parameters.

    z_face_color : ColorLike, optional
        Color used for the z-axis arrow.  Defaults to theme axes
        parameters.

    color_box : bool, optional
        Enable or disable the face colors.  Otherwise, box is white.

    label_color : ColorLike, optional
        Color of the labels.

    labels_off : bool, optional
        Enable or disable the text labels for the axes.

    opacity : float, optional
        Opacity in the range of ``[0, 1]`` of the orientation box.

    show_text_edges : bool, optional
        Enable or disable drawing the vector text edges.

    Returns
    -------
    :vtk:`vtkAnnotatedCubeActor`
        Annotated cube actor.

    Examples
    --------
    Create and plot an orientation box

    >>> import pyvista as pv
    >>> actor = pv.create_axes_orientation_box(
    ...     line_width=1,
    ...     text_scale=0.53,
    ...     edge_color='black',
    ...     x_color='k',
    ...     y_color=None,
    ...     z_color=None,
    ...     xlabel='X',
    ...     ylabel='Y',
    ...     zlabel='Z',
    ...     color_box=False,
    ...     labels_off=False,
    ...     opacity=1.0,
    ... )
    >>> pl = pv.Plotter()
    >>> _ = pl.add_actor(actor)
    >>> pl.show()

    """
    x_color = Color(x_color, default_color=pyvista.global_theme.axes.x_color)
    y_color = Color(y_color, default_color=pyvista.global_theme.axes.y_color)
    z_color = Color(z_color, default_color=pyvista.global_theme.axes.z_color)
    edge_color = Color(edge_color, default_color=pyvista.global_theme.edge_color)
    x_face_color = Color(x_face_color)
    y_face_color = Color(y_face_color)
    z_face_color = Color(z_face_color)
    axes_actor = _vtk.vtkAnnotatedCubeActor()
    axes_actor.SetFaceTextScale(text_scale)
    if xlabel is not None:
        axes_actor.SetXPlusFaceText(f'+{xlabel}')
        axes_actor.SetXMinusFaceText(f'-{xlabel}')
    if ylabel is not None:
        axes_actor.SetYPlusFaceText(f'+{ylabel}')
        axes_actor.SetYMinusFaceText(f'-{ylabel}')
    if zlabel is not None:
        axes_actor.SetZPlusFaceText(f'+{zlabel}')
        axes_actor.SetZMinusFaceText(f'-{zlabel}')
    axes_actor.SetFaceTextVisibility(not labels_off)
    axes_actor.SetTextEdgesVisibility(show_text_edges)
    # https://github.com/pyvista/pyvista/pull/5382
    # axes_actor.GetTextEdgesProperty().SetColor(edge_color.float_rgb)
    axes_actor.GetTextEdgesProperty().SetLineWidth(line_width)
    axes_actor.GetXPlusFaceProperty().SetColor(x_color.float_rgb)
    axes_actor.GetXMinusFaceProperty().SetColor(x_color.float_rgb)
    axes_actor.GetYPlusFaceProperty().SetColor(y_color.float_rgb)
    axes_actor.GetYMinusFaceProperty().SetColor(y_color.float_rgb)
    axes_actor.GetZPlusFaceProperty().SetColor(z_color.float_rgb)
    axes_actor.GetZMinusFaceProperty().SetColor(z_color.float_rgb)

    axes_actor.GetCubeProperty().SetOpacity(opacity)
    axes_actor.GetCubeProperty().SetEdgeColor(edge_color.float_rgb)
    axes_actor.GetCubeProperty().SetEdgeVisibility(True)
    axes_actor.GetCubeProperty().BackfaceCullingOn()
    if opacity < 1.0:
        # Hide the text edges
        axes_actor.GetTextEdgesProperty().SetOpacity(0)

    if color_box:
        # Hide the cube so we can color each face
        axes_actor.GetCubeProperty().SetOpacity(0)
        axes_actor.GetCubeProperty().SetEdgeVisibility(False)

        cube = pyvista.Cube()
        cube.clear_data()  # remove normals
        face_colors = np.array(
            [
                x_face_color.int_rgb,
                x_face_color.int_rgb,
                y_face_color.int_rgb,
                y_face_color.int_rgb,
                z_face_color.int_rgb,
                z_face_color.int_rgb,
            ],
            np.uint8,
        )
        cube.cell_data['face_colors'] = face_colors

        cube_mapper = _vtk.vtkPolyDataMapper()
        cube_mapper.SetInputData(cube)
        cube_mapper.SetColorModeToDirectScalars()
        cube_mapper.Update()

        cube_actor = pyvista.Actor(mapper=cube_mapper)
        cube_actor.prop.culling = 'back'
        cube_actor.prop.opacity = opacity

        prop_assembly = _vtk.vtkPropAssembly()
        prop_assembly.AddPart(axes_actor)
        prop_assembly.AddPart(cube_actor)
        actor = prop_assembly
    else:
        actor = axes_actor  # type: ignore[assignment]

    _update_axes_label_color(actor, label_color)

    return actor


def create_north_arrow():
    """Create a north arrow mesh.

    .. versionadded:: 0.44.0

    Returns
    -------
    pyvista.PolyData
        North arrow mesh.

    """
    points = np.array(
        [
            [0.0, 5.0, 0.0],
            [-2.0, 0.0, 0.0],
            [0.0, 1.5, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 5.0, 1.0],
            [-2.0, 0.0, 1.0],
            [0.0, 1.5, 1.0],
            [2.0, 0.0, 1.0],
        ],
    )
    faces = np.array(
        [
            4,
            3,
            7,
            4,
            0,
            4,
            2,
            6,
            7,
            3,
            4,
            1,
            5,
            6,
            2,
            4,
            0,
            4,
            5,
            1,
            4,
            0,
            1,
            2,
            3,
            4,
            4,
            7,
            6,
            5,
        ],
    )
    return pyvista.PolyData(points, faces)


def normalize(x, minimum=None, maximum=None):
    """Normalize the given value between [minimum, maximum].

    Parameters
    ----------
    x : numpy.ndarray
        The array of values to normalize.
    minimum : float, optional
        The minimum value to which ``x`` should be normalized. If not specified,
        the minimum value in ``x`` will be used.
    maximum : float, optional
        The maximum value to which ``x`` should be normalized. If not specified,
        the maximum value in ``x`` will be used.

    Returns
    -------
    numpy.ndarray
        The normalized array of values, where the values are scaled to the
        range ``[minimum, maximum]``.

    """
    if minimum is None:
        minimum = np.nanmin(x)
    if maximum is None:
        maximum = np.nanmax(x)
    return (x - minimum) / (maximum - minimum)


@_deprecate_positional_args(allowed=['mapping', 'n_colors'])
def opacity_transfer_function(  # noqa: PLR0917
    mapping,
    n_colors,
    interpolate: bool = True,  # noqa: FBT001, FBT002
    kind='linear',
):
    """Get the opacity transfer function for a mapping.

    These values will map on to a scalar bar range and thus the number of
    colors (``n_colors``) must correspond to the number of colors in the color
    mapping that these opacities are associated to.

    If interpolating, ``scipy.interpolate.interp1d`` is used if available,
    otherwise ``np.interp`` is used. The ``kind`` argument controls the kind of
    interpolation for ``interp1d``.

    This returns the opacity range from 0 to 255, where 0 is totally
    transparent and 255 is totally opaque.

    The equation to create the sigmoid mapping is: ``1 / (1 + exp(-x))`` where
    ``x`` is the range from ``-a`` to ``+a`` and ``a`` is the value given in
    the ``mapping`` string. Default is ``a=10`` for 'sigmoid' mapping.

    Parameters
    ----------
    mapping : list[float] | str
        The opacity mapping to use. Can be a ``str`` name of a predefined
        mapping including ``'linear'``, ``'geom'``, ``'sigmoid'``,
        ``'sigmoid_1-10,15,20'``, and ``foreground``. Append an ``'_r'`` to any
        of those names (except ``foreground``) to reverse that mapping.
        The mapping can also be a custom user-defined array/list of values
        that will be interpolated across the ``n_color`` range.

    n_colors : int
        The number of colors that the opacities must be mapped to.

    interpolate : bool
        Flag on whether or not to interpolate the opacity mapping for all
        colors.

    kind : str
        The interpolation kind if ``interpolate`` is ``True`` and ``scipy``
        is available. If ``scipy`` is not available, linear interpolation
        is always used. Options are:

        - ``'linear'``
        - ``'nearest'``
        - ``'zero'``
        - ``'slinear'``
        - ``'quadratic'``
        - ``'cubic'``
        - ``'previous'``
        - ``'next'``

        .. versionchanged:: 0.46

            Linear interpolation is now always used by default. Previously,
            quadratic interpolation was used if ``scipy`` was installed.

    Returns
    -------
    numpy.ndarray
        Array of ``numpy.uint8`` values ``n_colors`` long containing the
        [0-255] opacity mapping values.

    Examples
    --------
    >>> import pyvista as pv
    >>> # Fetch the `sigmoid` mapping between 0 and 255
    >>> tf = pv.opacity_transfer_function('sigmoid', 256)
    >>> # Fetch the `geom_r` mapping between 0 and 1
    >>> tf = pv.opacity_transfer_function('geom_r', 256).astype(float) / 255.0
    >>> # Interpolate a user defined opacity mapping
    >>> opacity = [0, 0.2, 0.9, 0.6, 0.3]
    >>> tf = pv.opacity_transfer_function(opacity, 256)

    """
    sigmoid = lambda x: np.array(1 / (1 + np.exp(-x)) * 255, dtype=np.uint8)
    transfer_func = {
        'linear': np.linspace(0, 255, n_colors, dtype=np.uint8),
        'geom': np.geomspace(1e-6, 255, n_colors, dtype=np.uint8),
        'geom_r': np.geomspace(255, 1e-6, n_colors, dtype=np.uint8),
        'sigmoid': sigmoid(np.linspace(-10.0, 10.0, n_colors)),
        'sigmoid_1': sigmoid(np.linspace(-1.0, 1.0, n_colors)),
        'sigmoid_2': sigmoid(np.linspace(-2.0, 2.0, n_colors)),
        'sigmoid_3': sigmoid(np.linspace(-3.0, 3.0, n_colors)),
        'sigmoid_4': sigmoid(np.linspace(-4.0, 4.0, n_colors)),
        'sigmoid_5': sigmoid(np.linspace(-5.0, 5.0, n_colors)),
        'sigmoid_6': sigmoid(np.linspace(-6.0, 6.0, n_colors)),
        'sigmoid_7': sigmoid(np.linspace(-7.0, 7.0, n_colors)),
        'sigmoid_8': sigmoid(np.linspace(-8.0, 8.0, n_colors)),
        'sigmoid_9': sigmoid(np.linspace(-9.0, 9.0, n_colors)),
        'sigmoid_10': sigmoid(np.linspace(-10.0, 10.0, n_colors)),
        'sigmoid_15': sigmoid(np.linspace(-15.0, 15.0, n_colors)),
        'sigmoid_20': sigmoid(np.linspace(-20.0, 20.0, n_colors)),
        'foreground': np.hstack((0, [255] * (n_colors - 1))).astype(np.uint8),
    }
    transfer_func['linear_r'] = transfer_func['linear'][::-1]
    transfer_func['sigmoid_r'] = transfer_func['sigmoid'][::-1]
    for i in range(3, 11):
        k = f'sigmoid_{i}'
        rk = f'{k}_r'
        transfer_func[rk] = transfer_func[k][::-1]
    if isinstance(mapping, str):
        try:
            return transfer_func[mapping]
        except KeyError:
            msg = (
                f'Opacity transfer function ({mapping}) unknown. '
                f'Valid options: {list(transfer_func.keys())}'
            )
            raise ValueError(msg) from None
    elif isinstance(mapping, (np.ndarray, list, tuple)):
        mapping = np.array(mapping)
        if mapping.size == n_colors:
            # User could pass transfer function ready for lookup table
            pass
        elif mapping.size < n_colors:
            # User pass custom transfer function to be linearly interpolated
            if np.max(mapping) > 1.0 or np.min(mapping) < 0.0:
                mapping = normalize(mapping)
            # Interpolate transfer function to match lookup table
            xo = np.linspace(0, n_colors, len(mapping), dtype=np.int_)
            xx = np.linspace(0, n_colors, n_colors, dtype=np.int_)
            try:
                if not interpolate:
                    msg = 'No interpolation.'
                    raise ValueError(msg)
                from scipy.interpolate import interp1d  # noqa: PLC0415

                f = interp1d(xo, mapping, kind=kind)
                vals = f(xx)
                vals[vals < 0] = 0.0
                vals[vals > 1.0] = 1.0
                mapping = (vals * 255.0).astype(np.uint8)

            except (ImportError, ValueError):
                # Otherwise use simple linear interp
                mapping = (np.interp(xx, xo, mapping) * 255).astype(np.uint8)
        else:
            msg = (
                f'Transfer function cannot have more values than `n_colors`. '
                f'This has {mapping.size} elements'
            )
            raise RuntimeError(msg)
        return mapping
    msg = f'Transfer function type ({type(mapping)}) not understood'
    raise TypeError(msg)


def parse_font_family(font_family: str) -> int:
    """Check and validate the given font family name.

    Parameters
    ----------
    font_family : str
        Font family name to validate. Must be one of the font names defined in
        the ``FONTS`` enum class.

    Returns
    -------
    int
        Corresponding integer value of the valid font family name in the
        ``FONTS`` enum class.

    Raises
    ------
    ValueError
        If the font_family is not one of the defined font names in the ``FONTS``
        enum class.

    """
    font_family = font_family.lower()
    fonts = [font.name for font in FONTS]
    if font_family not in fonts:
        msg = f'Font must one of the following:\n{", ".join(fonts)}'
        raise ValueError(msg)
    return FONTS[font_family].value


def check_math_text_support() -> bool:  # pragma: no cover
    """Raise a DeprecationError as this has been moved.

    Returns
    -------
    bool
        Returns False for compatibility.

    """
    from pyvista.core.errors import DeprecationError  # noqa: PLC0415

    # Deprecated on v0.47.0, estimated removal on v0.50.0
    msg = '`check_math_text_support` is now imported from `pyvista.report`'
    DeprecationError(msg)

    return False


def check_matplotlib_vtk_compatibility() -> bool:  # pragma: no cover
    """Raise a DeprecationError as this has been moved.

    Returns
    -------
    bool
        Returns False for compatibility.

    """
    from pyvista.core.errors import DeprecationError  # noqa: PLC0415

    # Deprecated on v0.47.0, estimated removal on v0.50.0
    msg = '`check_matplotlib_vtk_compatibility` is now imported from `pyvista.report`'
    DeprecationError(msg)
    return False  # returning bool for compatibility
