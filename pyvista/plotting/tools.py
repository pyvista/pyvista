"""Module containing useful plotting tools."""

from collections.abc import Sequence
from enum import Enum
import os
import platform
from subprocess import PIPE, Popen

import numpy as np

import pyvista
from pyvista import _vtk

from .colors import string_to_rgb


class FONTS(Enum):
    """Font families available to PyVista."""

    arial = _vtk.VTK_ARIAL
    courier = _vtk.VTK_COURIER
    times = _vtk.VTK_TIMES


# Track render window support and plotting
SUPPORTS_OPENGL = None
SUPPORTS_PLOTTING = None


def supports_open_gl():
    """Return if the system supports OpenGL."""
    global SUPPORTS_OPENGL
    if SUPPORTS_OPENGL is None:
        ren_win = _vtk.vtkRenderWindow()
        SUPPORTS_OPENGL = bool(ren_win.SupportsOpenGL())
    return SUPPORTS_OPENGL


def _system_supports_plotting():
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
        proc = Popen(["pgrep", "-qx", "Finder"], stdout=PIPE, stderr=PIPE)
        proc.communicate()
        if proc.returncode == 0:
            return True

        # display variable set, likely available
        return 'DISPLAY' in os.environ

    # Linux case
    try:
        proc = Popen(["xset", "-q"], stdout=PIPE, stderr=PIPE)
        proc.communicate()
        return proc.returncode == 0
    except OSError:
        return False


def system_supports_plotting():
    """Check if the environment supports plotting.

    Returns
    -------
    bool
        ``True`` when system supports plotting.

    """
    global SUPPORTS_PLOTTING
    if SUPPORTS_PLOTTING is None:
        SUPPORTS_PLOTTING = _system_supports_plotting()

    # always use the cached response
    return SUPPORTS_PLOTTING


def _update_axes_label_color(axes_actor, color=None):
    """Set the axes label color (internal helper)."""
    if color is None:
        color = pyvista.global_theme.font.color
    color = parse_color(color)
    if isinstance(axes_actor, _vtk.vtkAxesActor):
        prop_x = axes_actor.GetXAxisCaptionActor2D().GetCaptionTextProperty()
        prop_y = axes_actor.GetYAxisCaptionActor2D().GetCaptionTextProperty()
        prop_z = axes_actor.GetZAxisCaptionActor2D().GetCaptionTextProperty()
        for prop in [prop_x, prop_y, prop_z]:
            prop.SetColor(color[0], color[1], color[2])
            prop.SetShadow(False)
    elif isinstance(axes_actor, _vtk.vtkAnnotatedCubeActor):
        axes_actor.GetTextEdgesProperty().SetColor(color)


def create_axes_marker(label_color=None, x_color=None, y_color=None,
                       z_color=None, xlabel='X', ylabel='Y', zlabel='Z',
                       labels_off=False, line_width=2):
    """Create an axis actor.

    Parameters
    ----------
    label_color : str or sequence, optional
        Unknown

    x_color : str or sequence, optional
        Color of the x axis text.

    y_color : str or sequence, optional
        Color of the y axis text.

    z_color : str or sequence, optional
        Color of the z axis text.

    xlabel : str, optional
        Text used for the x axis.

    ylabel : str, optional
        Text used for the y axis.

    zlabel : str, optional
        Text used for the z axis.

    labels_off : bool, optional
        Enable or disable the text labels for the axes.

    line_width : float, optional
        The width of the marker lines.

    Returns
    -------
    vtk.vtkAxesActor
        Axes actor.

    """
    if x_color is None:
        x_color = pyvista.global_theme.axes.x_color
    if y_color is None:
        y_color = pyvista.global_theme.axes.y_color
    if z_color is None:
        z_color = pyvista.global_theme.axes.z_color
    axes_actor = _vtk.vtkAxesActor()
    axes_actor.GetXAxisShaftProperty().SetColor(parse_color(x_color))
    axes_actor.GetXAxisTipProperty().SetColor(parse_color(x_color))
    axes_actor.GetYAxisShaftProperty().SetColor(parse_color(y_color))
    axes_actor.GetYAxisTipProperty().SetColor(parse_color(y_color))
    axes_actor.GetZAxisShaftProperty().SetColor(parse_color(z_color))
    axes_actor.GetZAxisTipProperty().SetColor(parse_color(z_color))
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

    _update_axes_label_color(axes_actor, label_color)

    return axes_actor


def create_axes_orientation_box(line_width=1, text_scale=0.366667,
                                edge_color='black', x_color=None,
                                y_color=None, z_color=None,
                                xlabel='X', ylabel='Y', zlabel='Z',
                                x_face_color='red',
                                y_face_color='green',
                                z_face_color='blue',
                                color_box=False, label_color=None,
                                labels_off=False, opacity=0.5):
    """Create a Box axes orientation widget with labels.

    Parameters
    ----------
    line_width : float, optional
        The width of the marker lines.

    text_scale : float, optional
        Size of the text relative to the faces.

    edge_color : str or sequence, optional
        Color of the edges.

    x_color : str or sequence, optional
        Color of the x axis text.

    y_color : str or sequence, optional
        Color of the y axis text.

    z_color : str or sequence, optional
        Color of the z axis text.

    xlabel : str, optional
        Text used for the x axis.

    ylabel : str, optional
        Text used for the y axis.

    zlabel : str, optional
        Text used for the z axis.

    x_face_color : str or sequence, optional
        Color used for the x axis arrow.  Defaults to theme axes
        parameters.

    y_face_color : str or sequence, optional
        Color used for the y axis arrow.  Defaults to theme axes
        parameters.

    z_face_color : str or sequence, optional
        Color used for the z axis arrow.  Defaults to theme axes
        parameters.

    color_box : bool, optional
        Enable or disable the face colors.  Otherwise, box is white.

    label_color : str or sequence, optional
        Color of the labels.

    labels_off : bool, optional
        Enable or disable the text labels for the axes.

    opacity : float, optional
        Opacity in the range of ``[0, 1]`` of the orientation box.

    Returns
    -------
    vtk.vtkAnnotatedCubeActor
        Annotated cube actor.

    Examples
    --------
    Create and plot an orientation box

    >>> import pyvista
    >>> actor = pyvista.create_axes_orientation_box(
    ...    line_width=1, text_scale=0.53,
    ...    edge_color='black', x_color='k',
    ...    y_color=None, z_color=None,
    ...    xlabel='X', ylabel='Y', zlabel='Z',
    ...    color_box=False,
    ...    labels_off=False, opacity=1.0)
    >>> pl = pyvista.Plotter()
    >>> _ = pl.add_actor(actor)
    >>> pl.show()

    """
    if x_color is None:
        x_color = pyvista.global_theme.axes.x_color
    if y_color is None:
        y_color = pyvista.global_theme.axes.y_color
    if z_color is None:
        z_color = pyvista.global_theme.axes.z_color
    if edge_color is None:
        edge_color = pyvista.global_theme.edge_color
    axes_actor = _vtk.vtkAnnotatedCubeActor()
    axes_actor.SetFaceTextScale(text_scale)
    if xlabel is not None:
        axes_actor.SetXPlusFaceText(f"+{xlabel}")
        axes_actor.SetXMinusFaceText(f"-{xlabel}")
    if ylabel is not None:
        axes_actor.SetYPlusFaceText(f"+{ylabel}")
        axes_actor.SetYMinusFaceText(f"-{ylabel}")
    if zlabel is not None:
        axes_actor.SetZPlusFaceText(f"+{zlabel}")
        axes_actor.SetZMinusFaceText(f"-{zlabel}")
    axes_actor.SetFaceTextVisibility(not labels_off)
    axes_actor.SetTextEdgesVisibility(False)
    # axes_actor.GetTextEdgesProperty().SetColor(parse_color(edge_color))
    # axes_actor.GetTextEdgesProperty().SetLineWidth(line_width)
    axes_actor.GetXPlusFaceProperty().SetColor(parse_color(x_color))
    axes_actor.GetXMinusFaceProperty().SetColor(parse_color(x_color))
    axes_actor.GetYPlusFaceProperty().SetColor(parse_color(y_color))
    axes_actor.GetYMinusFaceProperty().SetColor(parse_color(y_color))
    axes_actor.GetZPlusFaceProperty().SetColor(parse_color(z_color))
    axes_actor.GetZMinusFaceProperty().SetColor(parse_color(z_color))

    axes_actor.GetCubeProperty().SetOpacity(opacity)
    # axes_actor.GetCubeProperty().SetEdgeColor(parse_color(edge_color))
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
        cube.clear_data() # remove normals
        face_colors = np.array([parse_color(x_face_color),
                                parse_color(x_face_color),
                                parse_color(y_face_color),
                                parse_color(y_face_color),
                                parse_color(z_face_color),
                                parse_color(z_face_color),
                                ])
        face_colors = (face_colors * 255).astype(np.uint8)
        cube.cell_data['face_colors'] = face_colors

        cube_mapper = _vtk.vtkPolyDataMapper()
        cube_mapper.SetInputData(cube)
        cube_mapper.SetColorModeToDirectScalars()
        cube_mapper.Update()

        cube_actor = _vtk.vtkActor()
        cube_actor.SetMapper(cube_mapper)
        cube_actor.GetProperty().BackfaceCullingOn()
        cube_actor.GetProperty().SetOpacity(opacity)

        prop_assembly = _vtk.vtkPropAssembly()
        prop_assembly.AddPart(axes_actor)
        prop_assembly.AddPart(cube_actor)
        actor = prop_assembly
    else:
        actor = axes_actor

    _update_axes_label_color(actor, label_color)

    return actor


def normalize(x, minimum=None, maximum=None):
    """Normalize the given value between [minimum, maximum]."""
    if minimum is None:
        minimum = np.nanmin(x)
    if maximum is None:
        maximum = np.nanmax(x)
    return (x - minimum) / (maximum - minimum)


def opacity_transfer_function(mapping, n_colors, interpolate=True,
                              kind='quadratic'):
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
    mapping : list(float) or str
        The opacity mapping to use. Can be a ``str`` name of a predefined
        mapping including 'linear', 'geom', 'sigmoid', 'sigmoid_3-10'.
        Append an '_r' to any of those names to reverse that mapping.
        This can also be a custom array/list of values that will be
        interpolated across the ``n_color`` range for user defined mappings.

    n_colors : int
        The amount of colors that the opacities must be mapped to.

    interpolate : bool
        Flag on whether or not to interpolate the opacity mapping for all colors

    kind : str
        The interepolation kind if ``interpolate`` is true and ``scipy`` is
        available. Options are ('linear', 'nearest', 'zero', 'slinear',
        'quadratic', 'cubic', 'previous', 'next'.

    Examples
    --------
    >>> import pyvista as pv
    >>> # Fetch the `sigmoid` mapping between 0 and 255
    >>> tf = pv.opacity_transfer_function("sigmoid", 256)
    >>> # Fetch the `geom_r` mapping between 0 and 1
    >>> tf = pv.opacity_transfer_function("geom_r", 256).astype(float) / 255.
    >>> # Interpolate a user defined opacity mapping
    >>> opacity = [0, 0.2, 0.9, 0.6, 0.3]
    >>> tf = pv.opacity_transfer_function(opacity, 256)

    """
    sigmoid = lambda x: np.array(1 / (1 + np.exp(-x)) * 255, dtype=np.uint8)
    transfer_func = {
        'linear': np.linspace(0, 255, n_colors, dtype=np.uint8),
        'geom': np.geomspace(1e-6, 255, n_colors, dtype=np.uint8),
        'geom_r': np.geomspace(255, 1e-6, n_colors, dtype=np.uint8),
        'sigmoid': sigmoid(np.linspace(-10., 10., n_colors)),
        'sigmoid_3': sigmoid(np.linspace(-3., 3., n_colors)),
        'sigmoid_4': sigmoid(np.linspace(-4., 4., n_colors)),
        'sigmoid_5': sigmoid(np.linspace(-5., 5., n_colors)),
        'sigmoid_6': sigmoid(np.linspace(-6., 6., n_colors)),
        'sigmoid_7': sigmoid(np.linspace(-7., 7., n_colors)),
        'sigmoid_8': sigmoid(np.linspace(-8., 8., n_colors)),
        'sigmoid_9': sigmoid(np.linspace(-9., 9., n_colors)),
        'sigmoid_10': sigmoid(np.linspace(-10., 10., n_colors)),
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
            raise KeyError(f'opactiy transfer function ({mapping}) unknown.')
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
                    raise ValueError('No interpolation.')
                # Use a quadratic interp if scipy is available
                from scipy.interpolate import interp1d

                # quadratic has best/smoothest results
                f = interp1d(xo, mapping, kind=kind)
                vals = f(xx)
                vals[vals < 0] = 0.0
                vals[vals > 1.0] = 1.0
                mapping = (vals * 255.).astype(np.uint8)

            except (ImportError, ValueError):
                # Otherwise use simple linear interp
                mapping = (np.interp(xx, xo, mapping) * 255).astype(np.uint8)
        else:
            raise RuntimeError(f'Transfer function cannot have more values than `n_colors`. This has {mapping.size} elements')
        return mapping
    raise TypeError(f'Transfer function type ({type(mapping)}) not understood')


def parse_color(color, opacity=None, default_color=None):
    """Parse color into a VTK friendly RGB(A) tuple.

    If ``color`` is a sequence of RGBA floats, the ``opacity`` parameter
    is ignored.

    Values returned will be between 0 and 1.

    Parameters
    ----------
    color : str or sequence
        Either a string, RGB sequence, RGBA sequence, or hex color string.
        RGB(A) sequences should only contain values between 0 and 1.
        For example:

        * ``'white'``
        * ``'w'``
        * ``[1, 1, 1]``
        * ``[0.5, 1, 0.7, 1]``
        * ``'#FFFFFF'``

    opacity : float, optional
        Default opacity of the returned color. Used when ``color`` is
        not a length 4 RGBA sequence. Only opacities between 0 and 1
        are allowed.

    default_color : str or sequence, optional
        Default color to use when ``color`` is None.  If this value is
        ``None``, then defaults to the global theme color.  Format is
        identical to ``color``.

    Returns
    -------
    tuple
        Either a length 3 RGB sequence if opacity is unset, or an RGBA
        sequence when ``opacity`` is set or the input ``color`` is an
        RGBA sequence.

    Examples
    --------
    >>> import pyvista
    >>> pyvista.parse_color('blue')
    (0.0, 0.0, 1.0)

    >>> pyvista.parse_color('k')
    (0.0, 0.0, 0.0)

    >>> pyvista.parse_color('#FFFFFF')
    (1.0, 1.0, 1.0)

    >>> pyvista.parse_color((0.4, 0.3, 0.4, 1))
    (0.4, 0.3, 0.4, 1.0)

    """
    color_valid = True
    if color is None:
        if default_color is None:
            color = pyvista.global_theme.color
        else:
            color = default_color
    if isinstance(color, str):
        color = string_to_rgb(color)
    elif isinstance(color, (Sequence, np.ndarray)):
        try:
            color = np.asarray(color, dtype=np.float64)
            if color.ndim != 1 or color.size not in (3, 4) or not np.all((0 <= color) & (color <= 1)):
                color_valid = False
            elif len(color) == 4:
                opacity = color[3]
                color = color[:3]
        except ValueError:
            color_valid = False
    else:
        color_valid = False
    if not color_valid:
        raise ValueError("\n"
                         f"\tInvalid color input: ({color})\n"
                         "\tMust be string, rgb list, or hex color string.  For example:\n"
                         "\t\tcolor='white'\n"
                         "\t\tcolor='w'\n"
                         "\t\tcolor=[1, 1, 1]\n"
                         "\t\tcolor='#FFFFFF'")
    if opacity is not None:
        if isinstance(opacity, (float, int)) and 0 <= opacity <= 1:
            color = [color[0], color[1], color[2], float(opacity)]
        else:
            raise ValueError("\n"
                             f"\tInvalid opacity input: {opacity}\n"
                             "\tMust be a scalar value between 0 and 1.")
    return tuple(color)


def parse_font_family(font_family):
    """Check font name."""
    font_family = font_family.lower()
    fonts = [font.name for font in FONTS]
    if font_family not in fonts:
        raise ValueError('Font must one of the following:\n{", ".join(fonts)}')
    return FONTS[font_family].value
