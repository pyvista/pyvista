import os
from subprocess import PIPE, Popen

import numpy as np
import vtk

import pyvista

from .theme import parse_color

def system_supports_plotting():
    """
    Check if x server is running

    Returns
    -------
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


def create_axes_orientation_box(line_width=1, text_scale=0.366667,
                                edge_color='black', x_color='lightblue',
                                y_color='seagreen', z_color='tomato',
                                x_face_color=(255, 0, 0),
                                y_face_color=(0, 255, 0),
                                z_face_color=(0, 0, 255),
                                color_box=False):
    """Create a Box axes orientation widget with labels.
    """
    axes_actor = vtk.vtkAnnotatedCubeActor()
    axes_actor.SetFaceTextScale(text_scale)
    axes_actor.SetXPlusFaceText("X+")
    axes_actor.SetXMinusFaceText("X-")
    axes_actor.SetYPlusFaceText("Y+")
    axes_actor.SetYMinusFaceText("Y-")
    axes_actor.SetZPlusFaceText("Z+")
    axes_actor.SetZMinusFaceText("Z-")
    axes_actor.GetTextEdgesProperty().SetColor(parse_color(edge_color))
    axes_actor.GetTextEdgesProperty().SetLineWidth(line_width)
    axes_actor.GetXPlusFaceProperty().SetColor(parse_color(x_color))
    axes_actor.GetXMinusFaceProperty().SetColor(parse_color(x_color))
    axes_actor.GetYPlusFaceProperty().SetColor(parse_color(y_color))
    axes_actor.GetYMinusFaceProperty().SetColor(parse_color(y_color))
    axes_actor.GetZPlusFaceProperty().SetColor(parse_color(z_color))
    axes_actor.GetZMinusFaceProperty().SetColor(parse_color(z_color))

    if color_box:
        # Hide the cube so we can color each face
        axes_actor.GetCubeProperty().SetOpacity(0)

        cube = pyvista.Cube()
        cube.clear_arrays() # remove normals
        face_colors = np.array([x_face_color,
                                x_face_color,
                                y_face_color,
                                y_face_color,
                                z_face_color,
                                z_face_color,
                               ], dtype=np.uint8)

        cube.cell_arrays['face_colors'] = face_colors

        cube_mapper = vtk.vtkPolyDataMapper()
        cube_mapper.SetInputData(cube)
        cube_mapper.SetColorModeToDirectScalars()
        cube_mapper.Update()

        cube_actor = vtk.vtkActor()
        cube_actor.SetMapper(cube_mapper)
        cube_actor.GetProperty().BackfaceCullingOn()

        prop_assembly = vtk.vtkPropAssembly()
        prop_assembly.AddPart(axes_actor)
        prop_assembly.AddPart(cube_actor)
        actor = prop_assembly
    else:
        actor = axes_actor

    return actor


def normalize(x, minimum=None, maximum=None):
    if minimum is None:
        minimum = np.nanmin(x)
    if maximum is None:
        maximum = np.nanmax(x)
    return (x - minimum) / (maximum - minimum)


def opacity_transfer_function(mapping, n_colors, interpolate=True):
    """Get the opacity transfer function results: range from 0 to 255.
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
        k = 'sigmoid_{}'.format(i)
        rk = '{}_r'.format(k)
        transfer_func[rk] = transfer_func[k][::-1]
    if isinstance(mapping, str):
        try:
            return transfer_func[mapping]
        except KeyError:
            raise KeyError('opactiy transfer function ({}) unknown.'.format(mapping))
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
            xo = np.linspace(0, n_colors, len(mapping), dtype=np.int)
            xx = np.linspace(0, n_colors, n_colors, dtype=np.int)
            try:
                if not interpolate:
                    raise AssertionError('No interpolation.')
                # Use a quadratic interp if scipy is available
                from scipy.interpolate import interp1d
                # quadratic has best/smoothest results
                f = interp1d(xo, mapping, kind='quadratic')
                vals = f(xx)
                vals[vals<0] = 0.0
                vals[vals>1.0] = 1.0
                mapping = (vals * 255.).astype(np.uint8)
            except (ImportError, AssertionError):
                # Otherwise use simple linear interp
                mapping = (np.interp(xx, xo, mapping) * 255).astype(np.uint8)
        else:
            raise RuntimeError('Transfer function cannot have more values than `n_colors`. This has {} elements'.format(mapping.size))
        return mapping
    raise TypeError('Transfer function type ({}) not understood'.format(type(mapping)))
