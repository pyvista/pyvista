"""Module containing useful plotting tools."""

import os
from subprocess import PIPE, Popen

import numpy as np
import vtk

import pyvista
from .theme import parse_color, rcParams


def system_supports_plotting():
    """Check if x server is running.

    Return
    ------
    system_supports_plotting : bool
        True when on Linux and running an xserver.  Returns None when
        on a non-linux platform.

    """
    try:
        if os.environ['ALLOW_PLOTTING'].lower() == 'true':
            return True
    except KeyError:
        pass
    try:
        p = Popen(["xset", "-q"], stdout=PIPE, stderr=PIPE)
        p.communicate()
        return p.returncode == 0
    except:
        return False


def update_axes_label_color(axes_actor, color=None):
    """Set the axes label color (internale helper)."""
    if color is None:
        color = rcParams['font']['color']
    color = parse_color(color)
    if isinstance(axes_actor, vtk.vtkAxesActor):
        prop_x = axes_actor.GetXAxisCaptionActor2D().GetCaptionTextProperty()
        prop_y = axes_actor.GetYAxisCaptionActor2D().GetCaptionTextProperty()
        prop_z = axes_actor.GetZAxisCaptionActor2D().GetCaptionTextProperty()
        for prop in [prop_x, prop_y, prop_z]:
            prop.SetColor(color[0], color[1], color[2])
            prop.SetShadow(False)
    elif isinstance(axes_actor, vtk.vtkAnnotatedCubeActor):
        axes_actor.GetTextEdgesProperty().SetColor(color)

    return


def create_axes_marker(label_color=None, x_color=None, y_color=None,
                       z_color=None, xlabel='X', ylabel='Y', zlabel='Z',
                       labels_off=False, line_width=2):
    """Return an axis actor to add in the scene."""
    if x_color is None:
        x_color = rcParams['axes']['x_color']
    if y_color is None:
        y_color = rcParams['axes']['y_color']
    if z_color is None:
        z_color = rcParams['axes']['z_color']
    axes_actor = vtk.vtkAxesActor()
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

    update_axes_label_color(axes_actor, label_color)

    return axes_actor


def create_axes_orientation_box(line_width=1, text_scale=0.366667,
                                edge_color='black', x_color=None,
                                y_color=None, z_color=None,
                                xlabel='X', ylabel='Y', zlabel='Z',
                                x_face_color='red',
                                y_face_color='green',
                                z_face_color='blue',
                                color_box=False, label_color=None,
                                labels_off=False, opacity=0.5,):
    """Create a Box axes orientation widget with labels."""
    if x_color is None:
        x_color = rcParams['axes']['x_color']
    if y_color is None:
        y_color = rcParams['axes']['y_color']
    if z_color is None:
        z_color = rcParams['axes']['z_color']
    if edge_color is None:
        edge_color = rcParams['edge_color']
    axes_actor = vtk.vtkAnnotatedCubeActor()
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
        cube.clear_arrays() # remove normals
        face_colors = np.array([parse_color(x_face_color),
                                parse_color(x_face_color),
                                parse_color(y_face_color),
                                parse_color(y_face_color),
                                parse_color(z_face_color),
                                parse_color(z_face_color),
                                ])
        face_colors = (face_colors * 255).astype(np.uint8)
        cube.cell_arrays['face_colors'] = face_colors

        cube_mapper = vtk.vtkPolyDataMapper()
        cube_mapper.SetInputData(cube)
        cube_mapper.SetColorModeToDirectScalars()
        cube_mapper.Update()

        cube_actor = vtk.vtkActor()
        cube_actor.SetMapper(cube_mapper)
        cube_actor.GetProperty().BackfaceCullingOn()
        cube_actor.GetProperty().SetOpacity(opacity)

        prop_assembly = vtk.vtkPropAssembly()
        prop_assembly.AddPart(axes_actor)
        prop_assembly.AddPart(cube_actor)
        actor = prop_assembly
    else:
        actor = axes_actor

    update_axes_label_color(actor, label_color)

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

    Example
    -------
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
        'sigmoid': sigmoid(np.linspace(-10.,10., n_colors)),
        'sigmoid_3': sigmoid(np.linspace(-3.,3., n_colors)),
        'sigmoid_4': sigmoid(np.linspace(-4.,4., n_colors)),
        'sigmoid_5': sigmoid(np.linspace(-5.,5., n_colors)),
        'sigmoid_6': sigmoid(np.linspace(-6.,6., n_colors)),
        'sigmoid_7': sigmoid(np.linspace(-7.,7., n_colors)),
        'sigmoid_8': sigmoid(np.linspace(-8.,8., n_colors)),
        'sigmoid_9': sigmoid(np.linspace(-9.,9., n_colors)),
        'sigmoid_10': sigmoid(np.linspace(-10.,10., n_colors)),

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
