"""Module managing different plotting theme parameters."""

import os

import vtk

from .colors import string_to_rgb, PARAVIEW_BACKGROUND

MAX_N_COLOR_BARS = 10
FONT_KEYS = {'arial': vtk.VTK_ARIAL,
             'courier': vtk.VTK_COURIER,
             'times': vtk.VTK_TIMES}

rcParams = {
    'auto_close': True, # DANGER: set to False with extreme caution
    'background': [0.3, 0.3, 0.3],
    'camera': {
        'position': [1, 1, 1],
        'viewup': [0, 0, 1],
    },
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
    'depth_peeling': {
        'number_of_peels': 4,
        'occlusion_ratio': 0.0,
        'enabled': False,
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

DEFAULT_THEME = dict(rcParams)

def set_plot_theme(theme):
    """Set the plotting parameters to a predefined theme."""
    if theme.lower() in ['paraview', 'pv']:
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
    elif theme.lower() in ['default']:
        for k,v in DEFAULT_THEME.items():
            rcParams[k] = v


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
