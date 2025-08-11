"""Color module supporting plotting module.

Used code from matplotlib.colors.  Thanks for your work.
"""

# Necessary for autodoc_type_aliases to recognize the type aliases used in the signatures
# of methods defined in this module.
from __future__ import annotations

from colorsys import rgb_to_hls
import contextlib
import importlib
import inspect
from typing import Literal
from typing import get_args

from cycler import Cycler
from cycler import cycler

try:
    from matplotlib import colormaps
    from matplotlib import colors
except ImportError:  # pragma: no cover
    # typing for newer versions of matplotlib
    # in newer versions cm is a module
    from matplotlib import cm as colormaps  # type: ignore[assignment]
    from matplotlib import colors

from typing import TYPE_CHECKING
from typing import Any

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

import pyvista
from pyvista import _validation
from pyvista._deprecate_positional_args import _deprecate_positional_args
from pyvista.core.utilities.misc import _NoNewAttrMixin

from . import _vtk

if TYPE_CHECKING:
    from ._typing import ColorLike
    from ._typing import ColormapOptions

IPYGANY_MAP = {
    'reds': 'Reds',
    'spectral': 'Spectral',
}

_ALLOWED_COLOR_NAME_DELIMITERS = '_' + '-' + ' '
_REMOVE_DELIMITER_LOOKUP = str.maketrans('', '', _ALLOWED_COLOR_NAME_DELIMITERS)


def _format_color_name(name: str):
    """Format name as lower-case and remove delimiters."""
    return name.lower().translate(_REMOVE_DELIMITER_LOOKUP)


def _format_color_dict(colors: dict[str, str]):
    """Format name and hex value."""
    return {_format_color_name(n): h.lower() for n, h in colors.items()}


# Colors from the CSS standard. Matches matplotlib.colors.CSS4_COLORS
# but with synonyms removed
_CSS_COLORS = {
    'aliceblue': '#F0F8FF',
    'antiquewhite': '#FAEBD7',
    'aquamarine': '#7FFFD4',
    'azure': '#F0FFFF',
    'beige': '#F5F5DC',
    'bisque': '#FFE4C4',
    'black': '#000000',
    'blanchedalmond': '#FFEBCD',
    'blue': '#0000FF',
    'blueviolet': '#8A2BE2',
    'brown': '#A52A2A',
    'burlywood': '#DEB887',
    'cadetblue': '#5F9EA0',
    'chartreuse': '#7FFF00',
    'chocolate': '#D2691E',
    'coral': '#FF7F50',
    'cornflowerblue': '#6495ED',
    'cornsilk': '#FFF8DC',
    'crimson': '#DC143C',
    'cyan': '#00FFFF',
    'darkblue': '#00008B',
    'darkcyan': '#008B8B',
    'darkgoldenrod': '#B8860B',
    'darkgray': '#A9A9A9',
    'darkgreen': '#006400',
    'darkkhaki': '#BDB76B',
    'darkmagenta': '#8B008B',
    'darkolivegreen': '#556B2F',
    'darkorange': '#FF8C00',
    'darkorchid': '#9932CC',
    'darkred': '#8B0000',
    'darksalmon': '#E9967A',
    'darkseagreen': '#8FBC8F',
    'darkslateblue': '#483D8B',
    'darkslategray': '#2F4F4F',
    'darkturquoise': '#00CED1',
    'darkviolet': '#9400D3',
    'deeppink': '#FF1493',
    'deepskyblue': '#00BFFF',
    'dimgray': '#696969',
    'dodgerblue': '#1E90FF',
    'firebrick': '#B22222',
    'floralwhite': '#FFFAF0',
    'forestgreen': '#228B22',
    'gainsboro': '#DCDCDC',
    'ghostwhite': '#F8F8FF',
    'gold': '#FFD700',
    'goldenrod': '#DAA520',
    'gray': '#808080',
    'green': '#008000',
    'greenyellow': '#ADFF2F',
    'honeydew': '#F0FFF0',
    'hotpink': '#FF69B4',
    'indianred': '#CD5C5C',
    'indigo': '#4B0082',
    'ivory': '#FFFFF0',
    'khaki': '#F0E68C',
    'lavender': '#E6E6FA',
    'lavenderblush': '#FFF0F5',
    'lawngreen': '#7CFC00',
    'lemonchiffon': '#FFFACD',
    'lightblue': '#ADD8E6',
    'lightcoral': '#F08080',
    'lightcyan': '#E0FFFF',
    'lightgoldenrodyellow': '#FAFAD2',
    'lightgray': '#D3D3D3',
    'lightgreen': '#90EE90',
    'lightpink': '#FFB6C1',
    'lightsalmon': '#FFA07A',
    'lightseagreen': '#20B2AA',
    'lightskyblue': '#87CEFA',
    'lightslategray': '#778899',
    'lightsteelblue': '#B0C4DE',
    'lightyellow': '#FFFFE0',
    'lime': '#00FF00',
    'limegreen': '#32CD32',
    'linen': '#FAF0E6',
    'magenta': '#FF00FF',
    'maroon': '#800000',
    'mediumaquamarine': '#66CDAA',
    'mediumblue': '#0000CD',
    'mediumorchid': '#BA55D3',
    'mediumpurple': '#9370DB',
    'mediumseagreen': '#3CB371',
    'mediumslateblue': '#7B68EE',
    'mediumspringgreen': '#00FA9A',
    'mediumturquoise': '#48D1CC',
    'mediumvioletred': '#C71585',
    'midnightblue': '#191970',
    'mintcream': '#F5FFFA',
    'mistyrose': '#FFE4E1',
    'moccasin': '#FFE4B5',
    'navajowhite': '#FFDEAD',
    'navy': '#000080',
    'oldlace': '#FDF5E6',
    'olive': '#808000',
    'olivedrab': '#6B8E23',
    'orange': '#FFA500',
    'orangered': '#FF4500',
    'orchid': '#DA70D6',
    'palegoldenrod': '#EEE8AA',
    'palegreen': '#98FB98',
    'paleturquoise': '#AFEEEE',
    'palevioletred': '#DB7093',
    'papayawhip': '#FFEFD5',
    'peachpuff': '#FFDAB9',
    'peru': '#CD853F',
    'pink': '#FFC0CB',
    'plum': '#DDA0DD',
    'powderblue': '#B0E0E6',
    'purple': '#800080',
    'rebeccapurple': '#663399',
    'red': '#FF0000',
    'rosybrown': '#BC8F8F',
    'royalblue': '#4169E1',
    'saddlebrown': '#8B4513',
    'salmon': '#FA8072',
    'sandybrown': '#F4A460',
    'seagreen': '#2E8B57',
    'seashell': '#FFF5EE',
    'sienna': '#A0522D',
    'silver': '#C0C0C0',
    'skyblue': '#87CEEB',
    'slateblue': '#6A5ACD',
    'slategray': '#708090',
    'snow': '#FFFAFA',
    'springgreen': '#00FF7F',
    'steelblue': '#4682B4',
    'tan': '#D2B48C',
    'teal': '#008080',
    'thistle': '#D8BFD8',
    'tomato': '#FF6347',
    'turquoise': '#40E0D0',
    'violet': '#EE82EE',
    'wheat': '#F5DEB3',
    'white': '#FFFFFF',
    'whitesmoke': '#F5F5F5',
    'yellow': '#FFFF00',
    'yellowgreen': '#9ACD32',
}

# Tableau colors. Matches matplotlib.colors.TABLEAU_COLORS
_TABLEAU_COLORS = {
    'tab:blue': '#1f77b4',
    'tab:orange': '#ff7f0e',
    'tab:green': '#2ca02c',
    'tab:red': '#d62728',
    'tab:purple': '#9467bd',
    'tab:brown': '#8c564b',
    'tab:pink': '#e377c2',
    'tab:gray': '#7f7f7f',
    'tab:olive': '#bcbd22',
    'tab:cyan': '#17becf',
}
_PARAVIEW_COLORS = {'paraview_background': '#52576e'}

# Colors from https://htmlpreview.github.io/?https://github.com/Kitware/vtk-examples/blob/gh-pages/VTKNamedColorPatches.html
# The vtk colors are only partially supported:
# - VTK colors with the same name as CSS colors but different values are excluded
#   (i.e. the CSS colors take precedent)
# - Not all VTK synonyms are supported.
# - Colors with adjective suffixes are renamed to use a prefix instead
#   (e.g. 'green_pale' is renamed to 'pale_green'). This is done to keep VTK color
#   names consistent with CSS names. In most cases this altered color name is
#   supported directly by vtkNamedColors, but in some cases this technically is no
#   longer a valid named vtk color. See tests.
_VTK_COLORS = {
    'alizarin_crimson': '#e32636',
    'aureoline_yellow': '#ffa824',
    'banana': '#e3cf57',
    'brick': '#9c661f',
    'brown_madder': '#db2929',
    'brown_ochre': '#87421f',
    'burnt_sienna': '#8a360f',
    'burnt_umber': '#8a3324',
    'cadmium_lemon': '#ffe303',
    'cadmium_orange': '#ff6103',
    'cadmium_yellow': '#ff9912',
    'carrot': '#ed9121',
    'cerulean': '#05b8cc',
    'chrome_oxide_green': '#668014',
    'cinnabar_green': '#61b329',
    'cobalt': '#3d59ab',
    'cobalt_green': '#3d9140',
    'cold_grey': '#808a87',
    'deep_cadmium_red': '#e3170d',
    'deep_cobalt_violet': '#91219e',
    'deep_naples_yellow': '#ffa812',
    'deep_ochre': '#733d1a',
    'eggshell': '#fce6c9',
    'emerald_green': '#00c957',
    'english_red': '#d43d1a',
    'flesh': '#ff7d40',
    'flesh_ochre': '#ff5721',
    'geranium_lake': '#e31230',
    'gold_ochre': '#c77826',
    'greenish_umber': '#ff3d0d',
    'ivory_black': '#292421',
    'lamp_black': '#2e473b',
    'light_cadmium_red': '#ff030d',
    'light_cadmium_yellow': '#ffb00f',
    'light_slate_blue': '#8470ff',
    'light_viridian': '#6eff70',
    'madder_lake_deep': '#e32e30',
    'manganese_blue': '#03a89e',
    'mars_orange': '#964514',
    'mars_yellow': '#e3701a',
    'melon': '#e3a869',
    'mint': '#bdfcc9',
    'peacock': '#33a1c9',
    'permanent_green': '#0ac92b',
    'permanent_red_violet': '#db2645',
    'raspberry': '#872657',
    'raw_sienna': '#C76114',
    'raw_umber': '#734a12',
    'rose_madder': '#e33638',
    'sap_green': '#308014',
    'sepia': '#5e2612',
    'terre_verte': '#385e0f',
    'titanium_white': '#fcfff0',
    'turquoise_blue': '#00c78c',
    'ultramarine': '#120a8f',
    'ultramarine_violet': '#5c246e',
    'van_dyke_brown': '#5e2605',
    'venetian_red': '#d41a1f',
    'violet_red': '#d02090',
    'warm_grey': '#808069',
    'yellow_ochre': '#e38217',
    'zinc_white': '#fcf7ff',
}

hexcolors = _format_color_dict(_CSS_COLORS | _PARAVIEW_COLORS | _TABLEAU_COLORS | _VTK_COLORS)

color_names = {h: n for n, h in hexcolors.items()}

color_char_to_word = {
    'b': 'blue',
    'g': 'green',
    'r': 'red',
    'c': 'cyan',
    'm': 'magenta',
    'y': 'yellow',
    'k': 'black',
    'w': 'white',
}

_color_synonyms = {
    **color_char_to_word,
    'aqua': 'cyan',
    'darkgrey': 'darkgray',
    'darkslategrey': 'darkslategray',
    'dimgrey': 'dimgray',
    'fuchsia': 'magenta',
    'grey': 'gray',
    'lightgrey': 'lightgray',
    'lightslategrey': 'lightslategray',
    'pv': 'paraview_background',
    'paraview': 'paraview_background',
    'slategrey': 'slategray',
    'lightgoldenrod': 'lightgoldenrodyellow',
}

color_synonyms = {
    _format_color_name(syn): _format_color_name(name) for syn, name in _color_synonyms.items()
}

matplotlib_default_colors = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf',
]

COLOR_SCHEMES = {
    'spectrum': {
        'id': _vtk.vtkColorSeries.SPECTRUM,
        'descr': 'black, red, blue, green, purple, orange, brown',
    },
    'warm': {'id': _vtk.vtkColorSeries.WARM, 'descr': 'dark red → yellow'},
    'cool': {'id': _vtk.vtkColorSeries.COOL, 'descr': 'green → blue → purple'},
    'blues': {'id': _vtk.vtkColorSeries.BLUES, 'descr': 'Different shades of blue'},
    'wild_flower': {
        'id': _vtk.vtkColorSeries.WILD_FLOWER,
        'descr': 'blue → purple → pink',
    },
    'citrus': {'id': _vtk.vtkColorSeries.CITRUS, 'descr': 'green → yellow → orange'},
    'div_purple_orange11': {
        'id': _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_11,
        'descr': 'dark brown → white → dark purple',
    },
    'div_purple_orange10': {
        'id': _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_10,
        'descr': 'dark brown → white → dark purple',
    },
    'div_purple_orange9': {
        'id': _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_9,
        'descr': 'brown → white → purple',
    },
    'div_purple_orange8': {
        'id': _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_8,
        'descr': 'brown → white → purple',
    },
    'div_purple_orange7': {
        'id': _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_7,
        'descr': 'brown → white → purple',
    },
    'div_purple_orange6': {
        'id': _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_6,
        'descr': 'brown → white → purple',
    },
    'div_purple_orange5': {
        'id': _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_5,
        'descr': 'orange → white → purple',
    },
    'div_purple_orange4': {
        'id': _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_4,
        'descr': 'orange → white → purple',
    },
    'div_purple_orange3': {
        'id': _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_3,
        'descr': 'orange → white → purple',
    },
    'div_spectral11': {
        'id': _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_11,
        'descr': 'dark red → light yellow → dark blue',
    },
    'div_spectral10': {
        'id': _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_10,
        'descr': 'dark red → light yellow → dark blue',
    },
    'div_spectral9': {
        'id': _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_9,
        'descr': 'red → light yellow → blue',
    },
    'div_spectral8': {
        'id': _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_8,
        'descr': 'red → light yellow → blue',
    },
    'div_spectral7': {
        'id': _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_7,
        'descr': 'red → light yellow → blue',
    },
    'div_spectral6': {
        'id': _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_6,
        'descr': 'red → light yellow → blue',
    },
    'div_spectral5': {
        'id': _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_5,
        'descr': 'red → light yellow → blue',
    },
    'div_spectral4': {
        'id': _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_4,
        'descr': 'red → light yellow → blue',
    },
    'div_spectral3': {
        'id': _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_3,
        'descr': 'orange → light yellow → green',
    },
    'div_brown_blue_green11': {
        'id': _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_11,
        'descr': 'dark brown → white → dark blue-green',
    },
    'div_brown_blue_green10': {
        'id': _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_10,
        'descr': 'dark brown → white → dark blue-green',
    },
    'div_brown_blue_green9': {
        'id': _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_9,
        'descr': 'brown → white → blue-green',
    },
    'div_brown_blue_green8': {
        'id': _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_8,
        'descr': 'brown → white → blue-green',
    },
    'div_brown_blue_green7': {
        'id': _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_7,
        'descr': 'brown → white → blue-green',
    },
    'div_brown_blue_green6': {
        'id': _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_6,
        'descr': 'brown → white → blue-green',
    },
    'div_brown_blue_green5': {
        'id': _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_5,
        'descr': 'brown → white → blue-green',
    },
    'div_brown_blue_green4': {
        'id': _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_4,
        'descr': 'brown → white → blue-green',
    },
    'div_brown_blue_green3': {
        'id': _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_3,
        'descr': 'brown → white → blue-green',
    },
    'seq_blue_green9': {
        'id': _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_9,
        'descr': 'light blue → dark green',
    },
    'seq_blue_green8': {
        'id': _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_8,
        'descr': 'light blue → dark green',
    },
    'seq_blue_green7': {
        'id': _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_7,
        'descr': 'light blue → dark green',
    },
    'seq_blue_green6': {
        'id': _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_6,
        'descr': 'light blue → green',
    },
    'seq_blue_green5': {
        'id': _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_5,
        'descr': 'light blue → green',
    },
    'seq_blue_green4': {
        'id': _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_4,
        'descr': 'light blue → green',
    },
    'seq_blue_green3': {
        'id': _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_3,
        'descr': 'light blue → green',
    },
    'seq_yellow_orange_brown9': {
        'id': _vtk.vtkColorSeries.BREWER_SEQUENTIAL_YELLOW_ORANGE_BROWN_9,
        'descr': 'light yellow → orange → dark brown',
    },
    'seq_yellow_orange_brown8': {
        'id': _vtk.vtkColorSeries.BREWER_SEQUENTIAL_YELLOW_ORANGE_BROWN_8,
        'descr': 'light yellow → orange → brown',
    },
    'seq_yellow_orange_brown7': {
        'id': _vtk.vtkColorSeries.BREWER_SEQUENTIAL_YELLOW_ORANGE_BROWN_7,
        'descr': 'light yellow → orange → brown',
    },
    'seq_yellow_orange_brown6': {
        'id': _vtk.vtkColorSeries.BREWER_SEQUENTIAL_YELLOW_ORANGE_BROWN_6,
        'descr': 'light yellow → orange → brown',
    },
    'seq_yellow_orange_brown5': {
        'id': _vtk.vtkColorSeries.BREWER_SEQUENTIAL_YELLOW_ORANGE_BROWN_5,
        'descr': 'light yellow → orange → brown',
    },
    'seq_yellow_orange_brown4': {
        'id': _vtk.vtkColorSeries.BREWER_SEQUENTIAL_YELLOW_ORANGE_BROWN_4,
        'descr': 'light yellow → orange',
    },
    'seq_yellow_orange_brown3': {
        'id': _vtk.vtkColorSeries.BREWER_SEQUENTIAL_YELLOW_ORANGE_BROWN_3,
        'descr': 'light yellow → orange',
    },
    'seq_blue_purple9': {
        'id': _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_PURPLE_9,
        'descr': 'light blue → dark purple',
    },
    'seq_blue_purple8': {
        'id': _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_PURPLE_8,
        'descr': 'light blue → purple',
    },
    'seq_blue_purple7': {
        'id': _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_PURPLE_7,
        'descr': 'light blue → purple',
    },
    'seq_blue_purple6': {
        'id': _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_PURPLE_6,
        'descr': 'light blue → purple',
    },
    'seq_blue_purple5': {
        'id': _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_PURPLE_5,
        'descr': 'light blue → purple',
    },
    'seq_blue_purple4': {
        'id': _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_PURPLE_4,
        'descr': 'light blue → purple',
    },
    'seq_blue_purple3': {
        'id': _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_PURPLE_3,
        'descr': 'light blue → purple',
    },
    'qual_accent': {
        'id': _vtk.vtkColorSeries.BREWER_QUALITATIVE_ACCENT,
        'descr': 'pastel green, pastel purple, pastel orange, pastel yellow, blue, pink, '
        'brown, gray',
    },
    'qual_dark2': {
        'id': _vtk.vtkColorSeries.BREWER_QUALITATIVE_DARK2,
        'descr': 'darker shade of qual_set2',
    },
    'qual_set3': {
        'id': _vtk.vtkColorSeries.BREWER_QUALITATIVE_SET3,
        'descr': 'pastel colors: blue green, light yellow, dark purple, red, blue, orange, green, '
        'pink, gray, purple, light green, yellow',
    },
    'qual_set2': {
        'id': _vtk.vtkColorSeries.BREWER_QUALITATIVE_SET2,
        'descr': 'blue green, orange, purple, pink, green, yellow, brown, gray',
    },
    'qual_set1': {
        'id': _vtk.vtkColorSeries.BREWER_QUALITATIVE_SET1,
        'descr': 'red, blue, green, purple, orange, yellow, brown, pink, gray',
    },
    'qual_pastel2': {
        'id': _vtk.vtkColorSeries.BREWER_QUALITATIVE_PASTEL2,
        'descr': 'pastel shade of qual_set2',
    },
    'qual_pastel1': {
        'id': _vtk.vtkColorSeries.BREWER_QUALITATIVE_PASTEL1,
        'descr': 'pastel shade of qual_set1',
    },
    'qual_paired': {
        'id': _vtk.vtkColorSeries.BREWER_QUALITATIVE_PAIRED,
        'descr': 'light blue, blue, light green, green, light red, red, light orange, orange, '
        'light purple, purple, light yellow',
    },
    'custom': {'id': _vtk.vtkColorSeries.CUSTOM, 'descr': None},
}

SCHEME_NAMES = {
    scheme_info['id']: scheme_name  # type: ignore[index]
    for scheme_name, scheme_info in COLOR_SCHEMES.items()
}

# Define colormaps that require colorcet
# matches set(colorcet.cm.keys()) - set(mpl.colormaps)
_COLORCET_CMAPS_LITERAL = Literal[
    'CET_C1',
    'CET_C10',
    'CET_C10_r',
    'CET_C10s',
    'CET_C10s_r',
    'CET_C11',
    'CET_C11_r',
    'CET_C11s',
    'CET_C11s_r',
    'CET_C1_r',
    'CET_C1s',
    'CET_C1s_r',
    'CET_C2',
    'CET_C2_r',
    'CET_C2s',
    'CET_C2s_r',
    'CET_C3',
    'CET_C3_r',
    'CET_C3s',
    'CET_C3s_r',
    'CET_C4',
    'CET_C4_r',
    'CET_C4s',
    'CET_C4s_r',
    'CET_C5',
    'CET_C5_r',
    'CET_C5s',
    'CET_C5s_r',
    'CET_C6',
    'CET_C6_r',
    'CET_C6s',
    'CET_C6s_r',
    'CET_C7',
    'CET_C7_r',
    'CET_C7s',
    'CET_C7s_r',
    'CET_C8',
    'CET_C8_r',
    'CET_C8s',
    'CET_C8s_r',
    'CET_C9',
    'CET_C9_r',
    'CET_C9s',
    'CET_C9s_r',
    'CET_CBC1',
    'CET_CBC1_r',
    'CET_CBC2',
    'CET_CBC2_r',
    'CET_CBD1',
    'CET_CBD1_r',
    'CET_CBD2',
    'CET_CBD2_r',
    'CET_CBL1',
    'CET_CBL1_r',
    'CET_CBL2',
    'CET_CBL2_r',
    'CET_CBL3',
    'CET_CBL3_r',
    'CET_CBL4',
    'CET_CBL4_r',
    'CET_CBTC1',
    'CET_CBTC1_r',
    'CET_CBTC2',
    'CET_CBTC2_r',
    'CET_CBTD1',
    'CET_CBTD1_r',
    'CET_CBTL1',
    'CET_CBTL1_r',
    'CET_CBTL2',
    'CET_CBTL2_r',
    'CET_CBTL3',
    'CET_CBTL3_r',
    'CET_CBTL4',
    'CET_CBTL4_r',
    'CET_D1',
    'CET_D10',
    'CET_D10_r',
    'CET_D11',
    'CET_D11_r',
    'CET_D12',
    'CET_D12_r',
    'CET_D13',
    'CET_D13_r',
    'CET_D1A',
    'CET_D1A_r',
    'CET_D1_r',
    'CET_D2',
    'CET_D2_r',
    'CET_D3',
    'CET_D3_r',
    'CET_D4',
    'CET_D4_r',
    'CET_D6',
    'CET_D6_r',
    'CET_D7',
    'CET_D7_r',
    'CET_D8',
    'CET_D8_r',
    'CET_D9',
    'CET_D9_r',
    'CET_I1',
    'CET_I1_r',
    'CET_I2',
    'CET_I2_r',
    'CET_I3',
    'CET_I3_r',
    'CET_L1',
    'CET_L10',
    'CET_L10_r',
    'CET_L11',
    'CET_L11_r',
    'CET_L12',
    'CET_L12_r',
    'CET_L13',
    'CET_L13_r',
    'CET_L14',
    'CET_L14_r',
    'CET_L15',
    'CET_L15_r',
    'CET_L16',
    'CET_L16_r',
    'CET_L17',
    'CET_L17_r',
    'CET_L18',
    'CET_L18_r',
    'CET_L19',
    'CET_L19_r',
    'CET_L1_r',
    'CET_L2',
    'CET_L20',
    'CET_L20_r',
    'CET_L2_r',
    'CET_L3',
    'CET_L3_r',
    'CET_L4',
    'CET_L4_r',
    'CET_L5',
    'CET_L5_r',
    'CET_L6',
    'CET_L6_r',
    'CET_L7',
    'CET_L7_r',
    'CET_L8',
    'CET_L8_r',
    'CET_L9',
    'CET_L9_r',
    'CET_R1',
    'CET_R1_r',
    'CET_R2',
    'CET_R2_r',
    'CET_R3',
    'CET_R3_r',
    'CET_R4',
    'CET_R4_r',
    'bgy',
    'bgy_r',
    'bgyw',
    'bgyw_r',
    'bjy',
    'bjy_r',
    'bkr',
    'bkr_r',
    'bky',
    'bky_r',
    'blues',
    'blues_r',
    'bmw',
    'bmw_r',
    'bmy',
    'bmy_r',
    'bwy',
    'bwy_r',
    'circle_mgbm_67_c31',
    'circle_mgbm_67_c31_r',
    'circle_mgbm_67_c31_s25',
    'circle_mgbm_67_c31_s25_r',
    'colorwheel',
    'colorwheel_r',
    'cwr',
    'cwr_r',
    'cyclic_bgrmb_35_70_c75',
    'cyclic_bgrmb_35_70_c75_r',
    'cyclic_bgrmb_35_70_c75_s25',
    'cyclic_bgrmb_35_70_c75_s25_r',
    'cyclic_grey_15_85_c0',
    'cyclic_grey_15_85_c0_r',
    'cyclic_grey_15_85_c0_s25',
    'cyclic_grey_15_85_c0_s25_r',
    'cyclic_isoluminant',
    'cyclic_isoluminant_r',
    'cyclic_mrybm_35_75_c68',
    'cyclic_mrybm_35_75_c68_r',
    'cyclic_mrybm_35_75_c68_s25',
    'cyclic_mrybm_35_75_c68_s25_r',
    'cyclic_mybm_20_100_c48',
    'cyclic_mybm_20_100_c48_r',
    'cyclic_mybm_20_100_c48_s25',
    'cyclic_mybm_20_100_c48_s25_r',
    'cyclic_mygbm_30_95_c78',
    'cyclic_mygbm_30_95_c78_r',
    'cyclic_mygbm_30_95_c78_s25',
    'cyclic_mygbm_30_95_c78_s25_r',
    'cyclic_mygbm_50_90_c46',
    'cyclic_mygbm_50_90_c46_r',
    'cyclic_mygbm_50_90_c46_s25',
    'cyclic_mygbm_50_90_c46_s25_r',
    'cyclic_protanopic_deuteranopic_bwyk_16_96_c31',
    'cyclic_protanopic_deuteranopic_bwyk_16_96_c31_r',
    'cyclic_protanopic_deuteranopic_wywb_55_96_c33',
    'cyclic_protanopic_deuteranopic_wywb_55_96_c33_r',
    'cyclic_rygcbmr_50_90_c64',
    'cyclic_rygcbmr_50_90_c64_r',
    'cyclic_rygcbmr_50_90_c64_s25',
    'cyclic_rygcbmr_50_90_c64_s25_r',
    'cyclic_tritanopic_cwrk_40_100_c20',
    'cyclic_tritanopic_cwrk_40_100_c20_r',
    'cyclic_tritanopic_wrwc_70_100_c20',
    'cyclic_tritanopic_wrwc_70_100_c20_r',
    'cyclic_wrkbw_10_90_c43',
    'cyclic_wrkbw_10_90_c43_r',
    'cyclic_wrkbw_10_90_c43_s25',
    'cyclic_wrkbw_10_90_c43_s25_r',
    'cyclic_wrwbw_40_90_c42',
    'cyclic_wrwbw_40_90_c42_r',
    'cyclic_wrwbw_40_90_c42_s25',
    'cyclic_wrwbw_40_90_c42_s25_r',
    'cyclic_ymcgy_60_90_c67',
    'cyclic_ymcgy_60_90_c67_r',
    'cyclic_ymcgy_60_90_c67_s25',
    'cyclic_ymcgy_60_90_c67_s25_r',
    'dimgray',
    'dimgray_r',
    'diverging_bkr_55_10_c35',
    'diverging_bkr_55_10_c35_r',
    'diverging_bky_60_10_c30',
    'diverging_bky_60_10_c30_r',
    'diverging_bwg_20_95_c41',
    'diverging_bwg_20_95_c41_r',
    'diverging_bwr_20_95_c54',
    'diverging_bwr_20_95_c54_r',
    'diverging_bwr_40_95_c42',
    'diverging_bwr_40_95_c42_r',
    'diverging_bwr_55_98_c37',
    'diverging_bwr_55_98_c37_r',
    'diverging_cwm_80_100_c22',
    'diverging_cwm_80_100_c22_r',
    'diverging_gkr_60_10_c40',
    'diverging_gkr_60_10_c40_r',
    'diverging_gwr_55_95_c38',
    'diverging_gwr_55_95_c38_r',
    'diverging_gwv_55_95_c39',
    'diverging_gwv_55_95_c39_r',
    'diverging_isoluminant_cjm_75_c23',
    'diverging_isoluminant_cjm_75_c23_r',
    'diverging_isoluminant_cjm_75_c24',
    'diverging_isoluminant_cjm_75_c24_r',
    'diverging_isoluminant_cjo_70_c25',
    'diverging_isoluminant_cjo_70_c25_r',
    'diverging_linear_bjr_30_55_c53',
    'diverging_linear_bjr_30_55_c53_r',
    'diverging_linear_bjy_30_90_c45',
    'diverging_linear_bjy_30_90_c45_r',
    'diverging_linear_protanopic_deuteranopic_bjy_57_89_c34',
    'diverging_linear_protanopic_deuteranopic_bjy_57_89_c34_r',
    'diverging_protanopic_deuteranopic_bwy_60_95_c32',
    'diverging_protanopic_deuteranopic_bwy_60_95_c32_r',
    'diverging_rainbow_bgymr_45_85_c67',
    'diverging_rainbow_bgymr_45_85_c67_r',
    'diverging_tritanopic_cwr_75_98_c20',
    'diverging_tritanopic_cwr_75_98_c20_r',
    'fire',
    'fire_r',
    'glasbey',
    'glasbey_bw',
    'glasbey_bw_minc_20',
    'glasbey_bw_minc_20_hue_150_280',
    'glasbey_bw_minc_20_hue_150_280_r',
    'glasbey_bw_minc_20_hue_330_100',
    'glasbey_bw_minc_20_hue_330_100_r',
    'glasbey_bw_minc_20_maxl_70',
    'glasbey_bw_minc_20_maxl_70_r',
    'glasbey_bw_minc_20_minl_30',
    'glasbey_bw_minc_20_minl_30_r',
    'glasbey_bw_minc_20_r',
    'glasbey_bw_r',
    'glasbey_category10',
    'glasbey_category10_r',
    'glasbey_cool',
    'glasbey_cool_r',
    'glasbey_dark',
    'glasbey_dark_r',
    'glasbey_hv',
    'glasbey_hv_r',
    'glasbey_light',
    'glasbey_light_r',
    'glasbey_r',
    'glasbey_warm',
    'glasbey_warm_r',
    'gouldian',
    'gouldian_r',
    'gwv',
    'gwv_r',
    'isolum',
    'isolum_r',
    'isoluminant_cgo_70_c39',
    'isoluminant_cgo_70_c39_r',
    'isoluminant_cgo_80_c38',
    'isoluminant_cgo_80_c38_r',
    'isoluminant_cm_70_c39',
    'isoluminant_cm_70_c39_r',
    'kb',
    'kb_r',
    'kbc',
    'kbc_r',
    'kbgyw',
    'kbgyw_r',
    'kg',
    'kg_r',
    'kgy',
    'kgy_r',
    'kr',
    'kr_r',
    'linear_bgy_10_95_c74',
    'linear_bgy_10_95_c74_r',
    'linear_bgyw_15_100_c67',
    'linear_bgyw_15_100_c67_r',
    'linear_bgyw_15_100_c68',
    'linear_bgyw_15_100_c68_r',
    'linear_bgyw_20_98_c66',
    'linear_bgyw_20_98_c66_r',
    'linear_blue_5_95_c73',
    'linear_blue_5_95_c73_r',
    'linear_blue_95_50_c20',
    'linear_blue_95_50_c20_r',
    'linear_bmw_5_95_c86',
    'linear_bmw_5_95_c86_r',
    'linear_bmw_5_95_c89',
    'linear_bmw_5_95_c89_r',
    'linear_bmy_10_95_c71',
    'linear_bmy_10_95_c71_r',
    'linear_bmy_10_95_c78',
    'linear_bmy_10_95_c78_r',
    'linear_gow_60_85_c27',
    'linear_gow_60_85_c27_r',
    'linear_gow_65_90_c35',
    'linear_gow_65_90_c35_r',
    'linear_green_5_95_c69',
    'linear_green_5_95_c69_r',
    'linear_grey_0_100_c0',
    'linear_grey_0_100_c0_r',
    'linear_grey_10_95_c0',
    'linear_grey_10_95_c0_r',
    'linear_kbc_5_95_c73',
    'linear_kbc_5_95_c73_r',
    'linear_kbgoy_20_95_c57',
    'linear_kbgoy_20_95_c57_r',
    'linear_kbgyw_10_98_c63',
    'linear_kbgyw_10_98_c63_r',
    'linear_kbgyw_5_98_c62',
    'linear_kbgyw_5_98_c62_r',
    'linear_kgy_5_95_c69',
    'linear_kgy_5_95_c69_r',
    'linear_kry_0_97_c73',
    'linear_kry_0_97_c73_r',
    'linear_kry_5_95_c72',
    'linear_kry_5_95_c72_r',
    'linear_kry_5_98_c75',
    'linear_kry_5_98_c75_r',
    'linear_kryw_0_100_c71',
    'linear_kryw_0_100_c71_r',
    'linear_kryw_5_100_c64',
    'linear_kryw_5_100_c64_r',
    'linear_kryw_5_100_c67',
    'linear_kryw_5_100_c67_r',
    'linear_protanopic_deuteranopic_kbjyw_5_95_c25',
    'linear_protanopic_deuteranopic_kbjyw_5_95_c25_r',
    'linear_protanopic_deuteranopic_kbw_5_95_c34',
    'linear_protanopic_deuteranopic_kbw_5_95_c34_r',
    'linear_protanopic_deuteranopic_kbw_5_98_c40',
    'linear_protanopic_deuteranopic_kbw_5_98_c40_r',
    'linear_protanopic_deuteranopic_kyw_5_95_c49',
    'linear_protanopic_deuteranopic_kyw_5_95_c49_r',
    'linear_ternary_blue_0_44_c57',
    'linear_ternary_blue_0_44_c57_r',
    'linear_ternary_green_0_46_c42',
    'linear_ternary_green_0_46_c42_r',
    'linear_ternary_red_0_50_c52',
    'linear_ternary_red_0_50_c52_r',
    'linear_tritanopic_kcw_5_95_c22',
    'linear_tritanopic_kcw_5_95_c22_r',
    'linear_tritanopic_krjcw_5_95_c24',
    'linear_tritanopic_krjcw_5_95_c24_r',
    'linear_tritanopic_krjcw_5_98_c46',
    'linear_tritanopic_krjcw_5_98_c46_r',
    'linear_tritanopic_krw_5_95_c46',
    'linear_tritanopic_krw_5_95_c46_r',
    'linear_wcmr_100_45_c42',
    'linear_wcmr_100_45_c42_r',
    'linear_worb_100_25_c53',
    'linear_worb_100_25_c53_r',
    'linear_wyor_100_45_c55',
    'linear_wyor_100_45_c55_r',
    'rainbow4',
    'rainbow4_r',
    'rainbow_bgyr_10_90_c83',
    'rainbow_bgyr_10_90_c83_r',
    'rainbow_bgyr_35_85_c72',
    'rainbow_bgyr_35_85_c72_r',
    'rainbow_bgyr_35_85_c73',
    'rainbow_bgyr_35_85_c73_r',
    'rainbow_bgyrm_35_85_c69',
    'rainbow_bgyrm_35_85_c69_r',
    'rainbow_bgyrm_35_85_c71',
    'rainbow_bgyrm_35_85_c71_r',
]
_COLORCET_CMAPS = get_args(_COLORCET_CMAPS_LITERAL)

# Define colormaps that require cmocean
# matches set(cmocean.cm.cmap_d.keys()) - set(mpl.colormaps)
_CMOCEAN_CMAPS_LITERAL = Literal[
    'algae',
    'algae_i',
    'algae_i_r',
    'algae_r',
    'algae_r_i',
    'amp',
    'amp_i',
    'amp_i_r',
    'amp_r',
    'amp_r_i',
    'balance',
    'balance_i',
    'balance_i_r',
    'balance_r',
    'balance_r_i',
    'curl',
    'curl_i',
    'curl_i_r',
    'curl_r',
    'curl_r_i',
    'deep',
    'deep_i',
    'deep_i_r',
    'deep_r',
    'deep_r_i',
    'delta',
    'delta_i',
    'delta_i_r',
    'delta_r',
    'delta_r_i',
    'dense',
    'dense_i',
    'dense_i_r',
    'dense_r',
    'dense_r_i',
    'diff',
    'diff_i',
    'diff_i_r',
    'diff_r',
    'diff_r_i',
    'gray_i',
    'gray_i_r',
    'gray_r_i',
    'haline',
    'haline_i',
    'haline_i_r',
    'haline_r',
    'haline_r_i',
    'ice',
    'ice_i',
    'ice_i_r',
    'ice_r',
    'ice_r_i',
    'matter',
    'matter_i',
    'matter_i_r',
    'matter_r',
    'matter_r_i',
    'oxy',
    'oxy_i',
    'oxy_i_r',
    'oxy_r',
    'oxy_r_i',
    'phase',
    'phase_i',
    'phase_i_r',
    'phase_r',
    'phase_r_i',
    'rain',
    'rain_i',
    'rain_i_r',
    'rain_r',
    'rain_r_i',
    'solar',
    'solar_i',
    'solar_i_r',
    'solar_r',
    'solar_r_i',
    'speed',
    'speed_i',
    'speed_i_r',
    'speed_r',
    'speed_r_i',
    'tarn',
    'tarn_i',
    'tarn_i_r',
    'tarn_r',
    'tarn_r_i',
    'tempo',
    'tempo_i',
    'tempo_i_r',
    'tempo_r',
    'tempo_r_i',
    'thermal',
    'thermal_i',
    'thermal_i_r',
    'thermal_r',
    'thermal_r_i',
    'topo',
    'topo_i',
    'topo_i_r',
    'topo_r',
    'topo_r_i',
    'turbid',
    'turbid_i',
    'turbid_i_r',
    'turbid_r',
    'turbid_r_i',
]
_CMOCEAN_CMAPS = get_args(_CMOCEAN_CMAPS_LITERAL)

_CMCRAMERI_CMAPS_LITERAL = Literal[
    'acton',
    'actonS',
    'acton_r',
    'bam',
    'bamO',
    'bamO_r',
    'bam_r',
    'bamako',
    'bamakoS',
    'bamako_r',
    'batlow',
    'batlowK',
    'batlowKS',
    'batlowK_r',
    'batlowS',
    'batlowW',
    'batlowWS',
    'batlowW_r',
    'batlow_r',
    'bilbao',
    'bilbaoS',
    'bilbao_r',
    'broc',
    'brocO',
    'brocO_r',
    'broc_r',
    'buda',
    'budaS',
    'buda_r',
    'bukavu',
    'bukavu_r',
    'cork',
    'corkO',
    'corkO_r',
    'cork_r',
    'davos',
    'davosS',
    'davos_r',
    'devon',
    'devonS',
    'devon_r',
    'fes',
    'fes_r',
    'glasgow',
    'glasgowS',
    'glasgow_r',
    'grayC',
    'grayCS',
    'grayC_r',
    'hawaii',
    'hawaiiS',
    'hawaii_r',
    'imola',
    'imolaS',
    'imola_r',
    'lajolla',
    'lajollaS',
    'lajolla_r',
    'lapaz',
    'lapazS',
    'lapaz_r',
    'lipari',
    'lipariS',
    'lipari_r',
    'lisbon',
    'lisbon_r',
    'navia',
    'naviaS',
    'navia_r',
    'nuuk',
    'nuukS',
    'nuuk_r',
    'oleron',
    'oleron_r',
    'oslo',
    'osloS',
    'oslo_r',
    'roma',
    'romaO',
    'romaO_r',
    'roma_r',
    'tofino',
    'tofino_r',
    'tokyo',
    'tokyoS',
    'tokyo_r',
    'turku',
    'turkuS',
    'turku_r',
    'vik',
    'vikO',
    'vikO_r',
    'vik_r',
]
_CMCRAMERI_CMAPS = get_args(_CMCRAMERI_CMAPS_LITERAL)

_MATPLOTLIB_CMAPS_LITERAL = Literal[
    'Accent',
    'Accent_r',
    'Blues',
    'Blues_r',
    'BrBG',
    'BrBG_r',
    'BuGn',
    'BuGn_r',
    'BuPu',
    'BuPu_r',
    'CMRmap',
    'CMRmap_r',
    'Dark2',
    'Dark2_r',
    'GnBu',
    'GnBu_r',
    'Grays',
    'Grays_r',
    'Greens',
    'Greens_r',
    'Greys',
    'Greys_r',
    'OrRd',
    'OrRd_r',
    'Oranges',
    'Oranges_r',
    'PRGn',
    'PRGn_r',
    'Paired',
    'Paired_r',
    'Pastel1',
    'Pastel1_r',
    'Pastel2',
    'Pastel2_r',
    'PiYG',
    'PiYG_r',
    'PuBu',
    'PuBuGn',
    'PuBuGn_r',
    'PuBu_r',
    'PuOr',
    'PuOr_r',
    'PuRd',
    'PuRd_r',
    'Purples',
    'Purples_r',
    'RdBu',
    'RdBu_r',
    'RdGy',
    'RdGy_r',
    'RdPu',
    'RdPu_r',
    'RdYlBu',
    'RdYlBu_r',
    'RdYlGn',
    'RdYlGn_r',
    'Reds',
    'Reds_r',
    'Set1',
    'Set1_r',
    'Set2',
    'Set2_r',
    'Set3',
    'Set3_r',
    'Spectral',
    'Spectral_r',
    'Wistia',
    'Wistia_r',
    'YlGn',
    'YlGnBu',
    'YlGnBu_r',
    'YlGn_r',
    'YlOrBr',
    'YlOrBr_r',
    'YlOrRd',
    'YlOrRd_r',
    'afmhot',
    'afmhot_r',
    'autumn',
    'autumn_r',
    'berlin',
    'berlin_r',
    'binary',
    'binary_r',
    'bone',
    'bone_r',
    'brg',
    'brg_r',
    'bwr',
    'bwr_r',
    'cividis',
    'cividis_r',
    'cool',
    'cool_r',
    'coolwarm',
    'coolwarm_r',
    'copper',
    'copper_r',
    'cubehelix',
    'cubehelix_r',
    'flag',
    'flag_r',
    'gist_earth',
    'gist_earth_r',
    'gist_gray',
    'gist_gray_r',
    'gist_grey',
    'gist_grey_r',
    'gist_heat',
    'gist_heat_r',
    'gist_ncar',
    'gist_ncar_r',
    'gist_rainbow',
    'gist_rainbow_r',
    'gist_stern',
    'gist_stern_r',
    'gist_yarg',
    'gist_yarg_r',
    'gist_yerg',
    'gist_yerg_r',
    'gnuplot',
    'gnuplot2',
    'gnuplot2_r',
    'gnuplot_r',
    'gray',
    'gray_r',
    'grey',
    'grey_r',
    'hot',
    'hot_r',
    'hsv',
    'hsv_r',
    'inferno',
    'inferno_r',
    'jet',
    'jet_r',
    'magma',
    'magma_r',
    'managua',
    'managua_r',
    'nipy_spectral',
    'nipy_spectral_r',
    'ocean',
    'ocean_r',
    'pink',
    'pink_r',
    'plasma',
    'plasma_r',
    'prism',
    'prism_r',
    'rainbow',
    'rainbow_r',
    'seismic',
    'seismic_r',
    'spring',
    'spring_r',
    'summer',
    'summer_r',
    'tab10',
    'tab10_r',
    'tab20',
    'tab20_r',
    'tab20b',
    'tab20b_r',
    'tab20c',
    'tab20c_r',
    'terrain',
    'terrain_r',
    'turbo',
    'turbo_r',
    'twilight',
    'twilight_r',
    'twilight_shifted',
    'twilight_shifted_r',
    'vanimo',
    'vanimo_r',
    'viridis',
    'viridis_r',
    'winter',
    'winter_r',
]

_MATPLOTLIB_CMAPS = get_args(_MATPLOTLIB_CMAPS_LITERAL)


class Color(_NoNewAttrMixin):
    """Helper class to convert between different color representations used in the pyvista library.

    Many pyvista methods accept :data:`ColorLike` parameters. This helper class
    is used to convert such parameters to the necessary format, used by
    underlying (VTK) methods. Any color name (``str``), hex string (``str``)
    or RGB(A) sequence (``tuple``, ``list`` or ``numpy.ndarray`` of ``int``
    or ``float``) is considered a :data:`ColorLike` parameter and can be converted
    by this class.

    See :ref:`named_colors` for a list of supported colors.

    .. note:

        The internally used representation is an integer RGBA sequence (values
        between 0 and 255). This might however change in future releases.

    Parameters
    ----------
    color : ColorLike, optional
        Either a string, RGB sequence, RGBA sequence, or hex color string.
        RGB(A) sequences should either be provided as floats between 0 and 1
        or as ints between 0 and 255. Hex color strings can contain optional
        ``'#'`` or ``'0x'`` prefixes. If no opacity is provided, the
        ``default_opacity`` will be used. If ``color`` is ``None``, the
        ``default_color`` is used instead.
        The following examples all denote the color 'white':

        * ``'white'``
        * ``'w'``
        * ``[1.0, 1.0, 1.0]``
        * ``[255, 255, 255, 255]``
        * ``'#FFFFFF'``

    opacity : int | float | str, optional
        Opacity of the represented color. Overrides any opacity associated
        with the provided ``color``. Allowed opacities are floats between 0
        and 1, ints between 0 and 255 or hexadecimal strings of length 2
        (plus the length of the optional prefix).
        The following examples all denote a fully opaque color:

        * ``1.0``
        * ``255``
        * ``'#ff'``

    default_color : ColorLike, optional
        Default color to use when ``color`` is ``None``. If this value is
        ``None``, then defaults to the global theme color. Format is
        identical to ``color``.

    default_opacity : int | float | str, optional
        Default opacity of the represented color. Used when ``color``
        does not specify an opacity and ``opacity`` is ``None``. Format
        is identical to ``opacity``.

    Examples
    --------
    Create a transparent green color using a color name, float RGBA sequence,
    integer RGBA sequence and RGBA hexadecimal string.

    >>> import pyvista as pv
    >>> pv.Color('green', opacity=0.5)
    Color(name='green', hex='#00800080', opacity=128)
    >>> pv.Color([0.0, 0.5, 0.0, 0.5])
    Color(name='green', hex='#00800080', opacity=128)
    >>> pv.Color([0, 128, 0, 128])
    Color(name='green', hex='#00800080', opacity=128)
    >>> pv.Color('#00800080')
    Color(name='green', hex='#00800080', opacity=128)

    """

    # Supported names for each color channel.
    CHANNEL_NAMES = (
        {'red', 'r'},  # 0
        {'green', 'g'},  # 1
        {'blue', 'b'},  # 2
        {'alpha', 'a', 'opacity'},  # 3
    )

    @_deprecate_positional_args(allowed=['color', 'opacity'])
    def __init__(  # noqa: PLR0917
        self,
        color: ColorLike | None = None,
        opacity: float | str | None = None,
        default_color: ColorLike | None = None,
        default_opacity: float | str = 255,
    ):
        """Initialize new instance."""
        self._red, self._green, self._blue, self._opacity = 0, 0, 0, 0
        self._opacity = self.convert_color_channel(default_opacity)
        self._name = None

        # Use default color if no color is provided
        if color is None:
            color = pyvista.global_theme.color if default_color is None else default_color

        _validation.check_instance(
            color, (Color, str, dict, list, tuple, np.ndarray, _vtk.vtkColor3ub), name='color'
        )
        try:
            if isinstance(color, Color):
                # Create copy of color instance
                self._red, self._green, self._blue, self._opacity = color.int_rgba
            elif isinstance(color, str):
                # From named color or hex string
                self._from_str(color)
            elif isinstance(color, dict):
                # From dictionary
                self._from_dict(color)
            elif isinstance(color, (list, tuple, np.ndarray)):
                # From RGB(A) sequence
                self._from_rgba(color)
            elif isinstance(color, _vtk.vtkColor3ub):
                # From vtkColor3ub instance (can be unpacked as rgb tuple)
                self._from_rgba(color)
            else:  # pragma: no cover
                msg = f'Unexpected color type: {type(color)}'
                raise TypeError(msg)
            self._name = color_names.get(self.hex_rgb, None)
        except ValueError as e:
            msg = (
                '\n'
                f'\tInvalid color input: ({color})\n'
                '\tMust be a string, rgb(a) sequence, or hex color string.  For example:\n'
                "\t\tcolor='white'\n"
                "\t\tcolor='w'\n"
                '\t\tcolor=[1.0, 1.0, 1.0]\n'
                '\t\tcolor=[255, 255, 255]\n'
                "\t\tcolor='#FFFFFF'"
            )
            raise ValueError(msg) from e

        # Overwrite opacity if it is provided
        try:
            if opacity is not None:
                self._opacity = self.convert_color_channel(opacity)
        except ValueError as e:
            msg = (
                '\n'
                f'\tInvalid opacity input: ({opacity})'
                '\tMust be an integer, float or string.  For example:\n'
                "\t\topacity='1.0'\n"
                "\t\topacity='255'\n"
                "\t\topacity='#FF'"
            )
            raise ValueError(msg) from e

    @staticmethod
    def strip_hex_prefix(h: str) -> str:
        """Strip any ``'#'`` or ``'0x'`` prefix from a hexadecimal string.

        Parameters
        ----------
        h : str
            Hexadecimal string to strip.

        Returns
        -------
        str
            Stripped hexadecimal string.

        """
        h = h.lstrip('#')
        return h.removeprefix('0x')

    @staticmethod
    def convert_color_channel(
        val: float | np.floating[Any] | np.integer[Any] | str,
    ) -> int:
        """Convert the given color channel value to the integer representation.

        Parameters
        ----------
        val : int | float | str
            Color channel value to convert. Supported input values are a
            hex string of length 2 (``'00'`` to ``'ff'``) with an optional
            prefix (``'#'`` or ``'0x'``), a float (``0.0`` to ``1.0``) or
            an integer (``0`` to ``255``).

        Returns
        -------
        int
            Color channel value in the integer representation (values between
            ``0`` and ``255``).

        """
        # Check for numpy inputs to avoid unnecessary calls to np.issubdtype
        arr = None
        if isinstance(val, (np.ndarray, np.generic)):
            arr = np.asanyarray(val)

        # Convert non-integers to int
        if isinstance(val, str):
            # From hexadecimal value
            val = int(Color.strip_hex_prefix(val), 16)
        elif isinstance(val, float) or (
            arr is not None and np.issubdtype(arr.dtype, np.floating) and arr.ndim == 0
        ):
            val = round(255 * val)

        # Check integers
        if isinstance(val, int) and 0 <= val <= 255:
            return val  # type: ignore[return-value]
        elif isinstance(val, np.uint8) or (
            arr is not None
            and np.issubdtype(arr.dtype, np.integer)
            and arr.ndim == 0
            and 0 <= val <= 255
        ):
            return int(val)
        else:
            msg = f'Unsupported color channel value provided: {val}'
            raise ValueError(msg)

    def _from_rgba(self, rgba):
        """Construct color from an RGB(A) sequence."""
        arg = rgba
        if len(rgba) == 3:
            # Keep using current opacity if it is not provided.
            rgba = [*rgba, self._opacity]
        try:
            if len(rgba) != 4:
                msg = 'Invalid length for RGBA sequence.'
                raise ValueError(msg)
            self._red, self._green, self._blue, self._opacity = (
                self.convert_color_channel(c) for c in rgba
            )
        except ValueError:
            msg = f'Invalid RGB(A) sequence: {arg}'
            raise ValueError(msg) from None

    def _from_dict(self, dct):
        """Construct color from an RGB(A) dictionary."""
        # Get any of the keys associated with each color channel (or None).
        rgba = [
            next((dct[key] for key in cnames if key in dct), None) for cnames in self.CHANNEL_NAMES
        ]
        self._from_rgba(rgba)

    def _from_hex(self, h):
        """Construct color from a hex string."""
        arg = h
        h = self.strip_hex_prefix(h)
        try:
            self._from_rgba(
                [self.convert_color_channel(h[i : i + 2]) for i in range(0, len(h), 2)]
            )
        except ValueError:
            msg = f'Invalid hex string: {arg}'
            raise ValueError(msg) from None

    def _from_str(self, n: str):
        """Construct color from a name or hex string."""
        arg = n
        n = _format_color_name(n)
        if n in color_synonyms:
            # Synonym of registered color name
            # Convert from synonym to full hex
            n = color_synonyms[n]
            self._from_hex(hexcolors[n])
        elif n in hexcolors:
            # Color name
            self._from_hex(hexcolors[n])
        else:
            # Otherwise, try conversion to hex
            try:
                self._from_hex(n)
            except ValueError:
                msg = f'Invalid color name or hex string: {arg}'
                raise ValueError(msg) from None

    @property
    def int_rgba(self) -> tuple[int, int, int, int]:  # numpydoc ignore=RT01
        """Get the color value as an RGBA integer tuple.

        Examples
        --------
        Create a blue color with half opacity.

        >>> import pyvista as pv
        >>> c = pv.Color('blue', opacity=128)
        >>> c
        Color(name='blue', hex='#0000ff80', opacity=128)
        >>> c.int_rgba
        (0, 0, 255, 128)

        Create a transparent red color using an integer RGBA sequence.

        >>> c = pv.Color([255, 0, 0, 64])
        >>> c
        Color(name='red', hex='#ff000040', opacity=64)
        >>> c.int_rgba
        (255, 0, 0, 64)

        """
        return self._red, self._green, self._blue, self._opacity

    @property
    def int_rgb(self) -> tuple[int, int, int]:  # numpydoc ignore=RT01
        """Get the color value as an RGB integer tuple.

        Examples
        --------
        Create a blue color with half opacity.

        >>> import pyvista as pv
        >>> c = pv.Color('blue', opacity=128)
        >>> c
        Color(name='blue', hex='#0000ff80', opacity=128)
        >>> c.int_rgb
        (0, 0, 255)

        Create a red color using an integer RGB sequence.

        >>> c = pv.Color([255, 0, 0])
        >>> c
        Color(name='red', hex='#ff0000ff', opacity=255)
        >>> c.int_rgb
        (255, 0, 0)

        """
        return self.int_rgba[:3]

    @property
    def float_rgba(self) -> tuple[float, float, float, float]:  # numpydoc ignore=RT01
        """Get the color value as an RGBA float tuple.

        Examples
        --------
        Create a blue color with custom opacity.

        >>> import pyvista as pv
        >>> c = pv.Color('blue', opacity=0.6)
        >>> c
        Color(name='blue', hex='#0000ff99', opacity=153)
        >>> c.float_rgba
        (0.0, 0.0, 1.0, 0.6)

        Create a transparent red color using a float RGBA sequence.

        >>> c = pv.Color([1.0, 0.0, 0.0, 0.2])
        >>> c
        Color(name='red', hex='#ff000033', opacity=51)
        >>> c.float_rgba
        (1.0, 0.0, 0.0, 0.2)

        """
        return (
            self._red / 255.0,
            self._green / 255.0,
            self._blue / 255.0,
            self._opacity / 255.0,
        )

    @property
    def float_rgb(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Get the color value as an RGB float tuple.

        Examples
        --------
        Create a blue color with custom opacity.

        >>> import pyvista as pv
        >>> c = pv.Color('blue', default_opacity=0.6)
        >>> c
        Color(name='blue', hex='#0000ff99', opacity=153)
        >>> c.float_rgb
        (0.0, 0.0, 1.0)

        Create a red color using a float RGB sequence.

        >>> c = pv.Color([1.0, 0.0, 0.0])
        >>> c
        Color(name='red', hex='#ff0000ff', opacity=255)
        >>> c.float_rgb
        (1.0, 0.0, 0.0)

        """
        return self.float_rgba[:3]

    @property
    def _float_hls(self) -> tuple[float, float, float]:
        """Get the color as Hue, Lightness, Saturation (HLS) in range [0.0, 1.0]."""
        return rgb_to_hls(*self.float_rgb)

    @property
    def hex_rgba(self) -> str:  # numpydoc ignore=RT01
        """Get the color value as an RGBA hexadecimal value.

        Examples
        --------
        Create a blue color with half opacity.

        >>> import pyvista as pv
        >>> c = pv.Color('blue', default_opacity='#80')
        >>> c
        Color(name='blue', hex='#0000ff80', opacity=128)
        >>> c.hex_rgba
        '#0000ff80'

        Create a transparent red color using an RGBA hexadecimal value.

        >>> c = pv.Color('0xff000040')
        >>> c
        Color(name='red', hex='#ff000040', opacity=64)
        >>> c.hex_rgba
        '#ff000040'

        """
        return '#' + ''.join(
            f'{c:0>2x}' for c in (self._red, self._green, self._blue, self._opacity)
        )

    @property
    def hex_rgb(self) -> str:  # numpydoc ignore=RT01
        """Get the color value as an RGB hexadecimal value.

        Examples
        --------
        Create a blue color with half opacity.

        >>> import pyvista as pv
        >>> c = pv.Color('blue', default_opacity='#80')
        >>> c
        Color(name='blue', hex='#0000ff80', opacity=128)
        >>> c.hex_rgb
        '#0000ff'

        Create a red color using an RGB hexadecimal value.

        >>> c = pv.Color('0xff0000')
        >>> c
        Color(name='red', hex='#ff0000ff', opacity=255)
        >>> c.hex_rgb
        '#ff0000'

        """
        return self.hex_rgba[:-2]

    @property
    def name(self) -> str | None:  # numpydoc ignore=RT01
        """Get the color name.

        The name is always formatted as a lower case string without
        any delimiters.

        See :ref:`named_colors` for a list of supported colors.

        Returns
        -------
        str | None
            The color name, in case this color has a name; otherwise ``None``.

        Examples
        --------
        Create a dark blue color with half opacity.

        >>> import pyvista as pv
        >>> c = pv.Color('darkblue', default_opacity=0.5)
        >>> c
        Color(name='darkblue', hex='#00008b80', opacity=128)

        When creating a new ``Color``, the name may be delimited with a space,
        hyphen, underscore, or written as a single word.

        >>> c = pv.Color('dark blue', default_opacity=0.5)

        Upper-case letters are also supported.

        >>> c = pv.Color('DarkBlue', default_opacity=0.5)

        However, the name is always standardized as a single lower-case word.

        >>> c
        Color(name='darkblue', hex='#00008b80', opacity=128)

        """
        return self._name

    @property
    def vtk_c3ub(self) -> _vtk.vtkColor3ub:  # numpydoc ignore=RT01
        """Get the color value as a VTK Color3ub instance.

        Examples
        --------
        Create a blue color with half opacity.

        >>> import pyvista as pv
        >>> c = pv.Color('blue', default_opacity=0.5)
        >>> c
        Color(name='blue', hex='#0000ff80', opacity=128)
        >>> c.vtk_c3ub
        vtkmodules.vtkCommonDataModel.vtkColor3ub([0, 0, 255])

        """
        return _vtk.vtkColor3ub(self._red, self._green, self._blue)

    def linear_to_srgb(self):
        """Convert from linear color values to sRGB color values.

        Returns
        -------
        Color
            A new ``Color`` instance with sRGB color values.

        """
        rgba = np.array(self.float_rgba)
        mask = rgba < 0.0031308
        rgba[mask] *= 12.92
        rgba[~mask] = 1.055 * rgba[~mask] ** (1 / 2.4) - 0.055
        return Color(rgba)

    def srgb_to_linear(self):
        """Convert from sRGB color values to linear color values.

        Returns
        -------
        Color
            A new ``Color`` instance with linear color values.

        """
        rgba = np.array(self.float_rgba)
        mask = rgba < 0.04045
        rgba[mask] /= 12.92
        rgba[~mask] = ((rgba[~mask] + 0.055) / 1.055) ** 2.4
        return Color(rgba)

    @classmethod
    def from_dict(cls, dict_):  # numpydoc ignore=RT01
        """Construct from dictionary for JSON deserialization."""
        return Color(dict_)

    def to_dict(self):  # numpydoc ignore=RT01
        """Convert to dictionary for JSON serialization."""
        return {'r': self._red, 'g': self._green, 'b': self._blue, 'a': self._opacity}

    @property
    def opacity(self):  # numpydoc ignore=RT01
        """Return the opacity of this color in the range of ``(0-255)``.

        Examples
        --------
        >>> import pyvista as pv
        >>> color = pv.Color('r', opacity=0.5)
        >>> color.opacity
        128
        >>> color
        Color(name='red', hex='#ff000080', opacity=128)

        """
        return self._opacity

    def __eq__(self, other):
        """Equality comparison."""
        try:
            return self.int_rgba == Color(other).int_rgba
        except ValueError:  # pragma: no cover
            return NotImplemented

    def __hash__(self):  # pragma: no cover
        """Hash calculation."""
        return hash((self._red, self._green, self._blue, self._opacity))

    def __getitem__(self, item):
        """Support indexing the float RGBA representation for backward compatibility."""
        if not isinstance(item, (str, slice, int, np.integer)):
            msg = 'Invalid index specified, only strings and integers are supported.'
            raise TypeError(msg)
        if isinstance(item, str):
            for i, cnames in enumerate(self.CHANNEL_NAMES):
                if item in cnames:
                    item = i
                    break
            else:
                msg = f'Invalid string index {item!r}.'
                raise ValueError(msg)
        return self.float_rgba[item]

    def __iter__(self):
        """Support iteration over the float RGBA representation for backward compatibility."""
        return iter(self.float_rgba)

    def __repr__(self) -> str:  # pragma: no cover
        """Human readable representation."""
        kwargs = f'hex={self.hex_rgba!r}, opacity={self.opacity}'
        if self._name is not None:
            kwargs = f'name={self._name!r}, ' + kwargs
        return f'Color({kwargs})'


PARAVIEW_BACKGROUND = Color('paraview').float_rgb  # [82, 87, 110] / 255


def get_cmap_safe(cmap: ColormapOptions) -> colors.Colormap:
    """Fetch a colormap by name from matplotlib, colorcet, cmocean, or cmcrameri.

    See :ref:`named_colormaps` for supported colormaps.

    Parameters
    ----------
    cmap : str | list[str] | matplotlib.colors.Colormap
        Name of the colormap to fetch. If the input is a list of strings, the
        strings must be color names (from :ref:`named_colors`), and
        it will create a ``ListedColormap`` with the input list.

    Returns
    -------
    matplotlib.colors.Colormap
        The requested colormap if available.

    Raises
    ------
    ValueError
        If the input colormap name is not valid.
    TypeError
        If the input is a list of items that are not strings.

    """
    _validation.check_instance(cmap, (str, list, colors.Colormap), name='cmap')

    def get_3rd_party_cmap(cmap_):
        cmap_sources = {
            'colorcet.cm': _COLORCET_CMAPS,
            'cmocean.cm.cmap_d': _CMOCEAN_CMAPS,
            'cmcrameri.cm.cmaps': _CMCRAMERI_CMAPS,
        }

        def get_nested_attr(obj, attr_path):
            for attr in attr_path:
                obj = getattr(obj, attr)
            return obj

        # Try importing and returning cmap from each package
        for cmap_import, known_cmaps in cmap_sources.items():
            parts = cmap_import.split('.')
            top_module = parts[0]

            with contextlib.suppress(ImportError):
                mod = importlib.import_module(top_module)
                cmap_dict = get_nested_attr(mod, parts[1:])
                with contextlib.suppress(KeyError):
                    return cmap_dict[cmap_]

            if cmap_ in known_cmaps:  # pragma: no cover
                msg = (
                    f'Package `{top_module}` is required to use colormap {cmap_!r}.\n'
                    'Install PyVista with `pyvista[colormaps]` to install it by default.'
                )
                raise ModuleNotFoundError(msg)
        return None

    if isinstance(cmap, colors.Colormap):
        return cmap
    if isinstance(cmap, str):
        # check if this colormap has been mapped between ipygany
        if cmap in IPYGANY_MAP:
            cmap = IPYGANY_MAP[cmap]  # type: ignore[assignment]

        cmap_3rd_party = get_3rd_party_cmap(cmap)
        if cmap_3rd_party:
            return cmap_3rd_party
        elif not isinstance(cmap, colors.Colormap):
            if inspect.ismodule(colormaps):  # pragma: no cover
                # Backwards compatibility with matplotlib<3.5.0
                if not hasattr(colormaps, cmap):
                    msg = f'Invalid colormap "{cmap}"'
                    raise ValueError(msg)
                cmap_obj = getattr(colormaps, cmap)
            else:
                try:
                    cmap_obj = colormaps[cmap]
                except KeyError:
                    msg = f"Invalid colormap '{cmap}'"
                    raise ValueError(msg) from None

    else:  # input is a list
        for item in cmap:
            if not isinstance(item, str):
                msg = 'When inputting a list as a cmap, each item should be a string.'  # type: ignore[unreachable]
                raise TypeError(msg)

        cmap_obj = ListedColormap(cmap)

    return cmap_obj


def get_default_cycler():
    """Return the default color cycler (matches matplotlib's default).

    Returns
    -------
    cycler.Cycler
        A cycler object for color that matches matplotlib's default colors.

    """
    return cycler('color', matplotlib_default_colors)


def get_hexcolors_cycler():
    """Return a color cycler for all of the available hexcolors.

    See ``pyvista.plotting.colors.hexcolors``.

    Returns
    -------
    cycler.Cycler
        A cycler object for color using all the available hexcolors from
        ``pyvista.plotting.colors.hexcolors``.

    """
    return cycler('color', hexcolors.keys())


def get_matplotlib_theme_cycler():
    """Return the color cycler of the current matplotlib theme.

    Returns
    -------
    cycler.Cycler
        Color cycler of the current matplotlib theme.

    """
    return plt.rcParams['axes.prop_cycle']


def color_scheme_to_cycler(scheme):
    """Convert a color scheme to a Cycler.

    Parameters
    ----------
    scheme : str | int | :vtk:`vtkColorSeries`
        Color scheme to be converted. If a string, it should correspond to a
        valid color scheme name (e.g., 'viridis'). If an integer, it should
        correspond to a valid color scheme ID. If an instance of
        :vtk:`vtkColorSeries`, it should be a valid color series.

    Returns
    -------
    cycler.Cycler
        A Cycler object with the color scheme.

    Raises
    ------
    ValueError
        If the provided `scheme` is not a valid color scheme.

    """
    if not isinstance(scheme, _vtk.vtkColorSeries):
        series = _vtk.vtkColorSeries()
        if isinstance(scheme, str):
            series.SetColorScheme(COLOR_SCHEMES.get(scheme.lower())['id'])  # type: ignore[index]
        elif isinstance(scheme, int):
            series.SetColorScheme(scheme)
        else:
            msg = f'Color scheme not understood: {scheme}'
            raise TypeError(msg)
    else:
        series = scheme
    colors = (series.GetColor(i) for i in range(series.GetNumberOfColors()))
    return cycler('color', colors)


def get_cycler(color_cycler):
    """Return a color cycler based on the input value.

    Parameters
    ----------
    color_cycler : str, list, tuple, or Cycler
        Specifies the desired color cycler. The value must be one of the following:
        - A list or tuple of color-like objects
        - A Cycler object with color-like objects
        - One of the following string values:
            - ``'default'``: Use the default color cycler (matches matplotlib's default)
            - ``'matplotlib'``: Dynamically get matplotlib's current theme's color cycler.
            - ``'all'``: Cycle through all available colors in
              ``pyvista.plotting.colors.hexcolors``
        - A named color scheme from ``pyvista.plotting.colors.COLOR_SCHEMES``

    Returns
    -------
    Cycler
        The color cycler corresponding to the input value.

    Raises
    ------
    ValueError
        Raised if the input is a string not found in named color schemes.
    TypeError
        Raised if the input is of an unsupported type.

    """
    if color_cycler is None:
        cycler_ = None
    elif isinstance(color_cycler, str):
        if color_cycler == 'default':
            cycler_ = get_default_cycler()
        elif color_cycler == 'matplotlib':
            cycler_ = get_matplotlib_theme_cycler()
        elif color_cycler == 'all':
            cycler_ = get_hexcolors_cycler()
        elif color_cycler in COLOR_SCHEMES:
            cycler_ = color_scheme_to_cycler(color_cycler)
        else:
            msg = f'color cycler of name `{color_cycler}` not found.'
            raise ValueError(msg)
    elif isinstance(color_cycler, (tuple, list)):
        cycler_ = cycler('color', color_cycler)
    elif isinstance(color_cycler, Cycler):
        cycler_ = color_cycler
    else:
        msg = f'color cycler of type {type(color_cycler)} not supported.'
        raise TypeError(msg)
    return cycler_
