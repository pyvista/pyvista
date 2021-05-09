"""Module managing different plotting theme parameters."""

import os

from pyvista import _vtk
from .colors import string_to_rgb, PARAVIEW_BACKGROUND

MAX_N_COLOR_BARS = 10
FONT_KEYS = {'arial': _vtk.VTK_ARIAL,
             'courier': _vtk.VTK_COURIER,
             'times': _vtk.VTK_TIMES}


def _load_default():
    """Generate the default theme.

    This is generated when called rather than provided statically to
    avoid parameters internal to the dictionary to being overwritten.
    These are not protected via ``dict(rcParams)`` as they do not copy
    internal lists.

    """
    return {
        'jupyter_backend': 'ipyvtklink',
        'auto_close': True,  # DANGER: set to False with extreme caution
        'background': [0.3, 0.3, 0.3],
        'full_screen': False,
        'camera': {
            'position': [1, 1, 1],
            'viewup': [0, 0, 1],
        },
        'notebook': None,
        'window_size': [1024, 768],
        'font': {
            'family': 'arial',
            'size': 12,
            'title_size': None,
            'label_size': None,
            'color': [1, 1, 1],
            'fmt': None,
        },
        'cmap': 'viridis',
        'color': 'white',
        'nan_color': 'darkgray',
        'edge_color': 'black',
        'outline_color': 'white',
        'floor_color': 'gray',
        'colorbar_orientation': 'horizontal',
        'colorbar_horizontal': {
            'width': 0.6,
            'height': 0.08,
            'position_x': 0.35,
            'position_y': 0.05,
        },
        'colorbar_vertical': {
            'width': 0.08,
            'height': 0.45,
            'position_x': 0.9,
            'position_y': 0.02,
        },
        'show_scalar_bar': True,
        'show_edges': False,
        'lighting': True,
        'interactive': False,
        'render_points_as_spheres': False,
        'use_ipyvtk': False,
        'transparent_background': False,
        'title': 'PyVista',
        'axes': {
            'x_color': 'tomato',
            'y_color': 'seagreen',
            'z_color': 'mediumblue',
            'box': False,
            'show': True,
        },
        'multi_samples': 4,
        'multi_rendering_splitting_position': None,
        'volume_mapper': 'fixed_point' if os.name == 'nt' else 'smart',
        'smooth_shading': False,
        'depth_peeling': {
            'number_of_peels': 4,
            'occlusion_ratio': 0.0,
            'enabled': False,
        },
        'silhouette': {
            'color': 'black',
            'line_width': 2,
            'opacity': 1.0,
            'feature_angle': False,
            'decimate': 0.9,
        },
        'slider_style': {
            'classic': {
                'slider_length': 0.02,
                'slider_width': 0.04,
                'slider_color': (0.5, 0.5, 0.5),
                'tube_width': 0.005,
                'tube_color': (1, 1, 1),
                'cap_opacity': 1,
                'cap_length': 0.01,
                'cap_width': 0.02,
            },
            'modern': {
                'slider_length': 0.02,
                'slider_width': 0.04,
                'slider_color': (0.43137255, 0.44313725, 0.45882353),
                'tube_width': 0.04,
                'tube_color': (0.69803922, 0.70196078, 0.70980392),
                'cap_opacity': 0,
                'cap_length': 0.01,
                'cap_width': 0.02,
            },
        },
    }


rcParams = _load_default()
DEFAULT_THEME = dict(rcParams)


def _reset_rcParams():
    """Reset rcParams in-place."""
    rcParams.clear()
    rcParams.update(_load_default())


def set_plot_theme(theme):
    """Set the plotting parameters to a predefined theme."""
    allowed_themes = ['paraview',
                      'pv',
                      'document',
                      'doc',
                      'paper',
                      'report',
                      'night',
                      'dark',
                      'testing',
                      'default']

    if theme.lower() in ['paraview', 'pv']:
        _reset_rcParams()
        rcParams['background'] = PARAVIEW_BACKGROUND
        rcParams['cmap'] = 'coolwarm'
        rcParams['font']['family'] = 'arial'
        rcParams['font']['label_size'] = 16
        rcParams['font']['color'] = 'white'
        rcParams['show_edges'] = False
        rcParams['color'] = 'white'
        rcParams['outline_color'] = 'white'
        rcParams['edge_color'] = 'black'
        rcParams['axes']['x_color'] = 'tomato'
        rcParams['axes']['y_color'] = 'gold'
        rcParams['axes']['z_color'] = 'green'
    elif theme.lower() in ['document', 'doc', 'paper', 'report']:
        _reset_rcParams()
        rcParams['background'] = 'white'
        rcParams['cmap'] = 'viridis'
        rcParams['font']['size'] = 18
        rcParams['font']['title_size'] = 18
        rcParams['font']['label_size'] = 18
        rcParams['font']['color'] = 'black'
        rcParams['show_edges'] = False
        rcParams['color'] = 'tan'
        rcParams['outline_color'] = 'black'
        rcParams['edge_color'] = 'black'
        rcParams['axes']['x_color'] = 'tomato'
        rcParams['axes']['y_color'] = 'seagreen'
        rcParams['axes']['z_color'] = 'blue'
    elif theme.lower() in ['night', 'dark']:
        _reset_rcParams()
        rcParams['background'] = 'black'
        rcParams['cmap'] = 'viridis'
        rcParams['font']['color'] = 'white'
        rcParams['show_edges'] = False
        rcParams['color'] = 'tan'
        rcParams['outline_color'] = 'white'
        rcParams['edge_color'] = 'white'
        rcParams['axes']['x_color'] = 'tomato'
        rcParams['axes']['y_color'] = 'seagreen'
        rcParams['axes']['z_color'] = 'blue'
    elif theme.lower() == 'testing':
        _reset_rcParams()
        # necessary for image regression.  Xvfb doesn't support
        # multi-sampling, so we disable it here for consistency between
        # desktops and remote testing
        rcParams['off_screen'] = True
        rcParams['multi_samples'] = 1
        rcParams['window_size'] = [400, 400]
    elif theme.lower() in ['default']:
        # have to clear and overwrite since some rcParams are not set
        # in the default theme
        _reset_rcParams()
    else:
        raise ValueError(f'Invalid theme {theme}.  Pick one of the following:\n'
                         f'{allowed_themes}')


def parse_color(color, opacity=None):
    """Parse color into a vtk friendly rgb list.

    Values returned will be between 0 and 1.

    """
    if color is None:
        color = rcParams['color']
    if isinstance(color, str):
        color = string_to_rgb(color)
    elif len(color) == 3:
        pass
    elif len(color) == 4:
        color = color[:3]
    else:
        raise ValueError(f"""
    Invalid color input: ({color})
    Must be string, rgb list, or hex color string.  For example:
        color='white'
        color='w'
        color=[1, 1, 1]
        color='#FFFFFF'""")
    if opacity is not None and isinstance(opacity, (float, int)):
        color = [color[0], color[1], color[2], opacity]
    return color


def parse_font_family(font_family):
    """Check font name."""
    # check font name
    font_family = font_family.lower()
    if font_family not in ['courier', 'times', 'arial']:
        raise ValueError('Font must be either "courier", "times" '
                         'or "arial"')

    return FONT_KEYS[font_family]
