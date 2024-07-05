"""Color module supporting plotting module.

Used code from matplotlib.colors.  Thanks for your work.


SUPPORTED COLORS
aliceblue
antiquewhite
aqua
aquamarine
azure
beige
bisque
black
blanchedalmond
blue
blueviolet
brown
burlywood
cadetblue
chartreuse
chocolate
coral
cornflowerblue
cornsilk
crimson
cyan
darkblue
darkcyan
darkgoldenrod
darkgray
darkgreen
darkgrey
darkkhaki
darkmagenta
darkolivegreen
darkorange
darkorchid
darkred
darksalmon
darkseagreen
darkslateblue
darkslategray
darkslategrey
darkturquoise
darkviolet
deeppink
deepskyblue
dimgray
dimgrey
dodgerblue
firebrick
floralwhite
forestgreen
fuchsia
gainsboro
ghostwhite
gold
goldenrod
gray
green
greenyellow
grey
honeydew
hotpink
indianred
indigo
ivory
khaki
lavender
lavenderblush
lawngreen
lemonchiffon
lightblue
lightcoral
lightcyan
lightgoldenrodyellow
lightgray
lightgreen
lightgrey
lightpink
lightsalmon
lightseagreen
lightskyblue
lightslategray
lightslategrey
lightsteelblue
lightyellow
lime
limegreen
linen
magenta
maroon
mediumaquamarine
mediumblue
mediumorchid
mediumpurple
mediumseagreen
mediumslateblue
mediumspringgreen
mediumturquoise
mediumvioletred
midnightblue
mintcream
mistyrose
moccasin
navajowhite
navy
oldlace
olive
olivedrab
orange
orangered
orchid
palegoldenrod
palegreen
paleturquoise
palevioletred
papayawhip
peachpuff
peru
pink
plum
powderblue
purple
rebeccapurple
red
rosybrown
royalblue
saddlebrown
salmon
sandybrown
seagreen
seashell
sienna
silver
skyblue
slateblue
slategray
slategrey
snow
springgreen
steelblue
tan
teal
thistle
tomato
turquoise
violet
wheat
white
whitesmoke
yellow
yellowgreen
tab:blue
tab:orange
tab:green
tab:red
tab:purple
tab:brown
tab:pink
tab:gray
tab:olive
tab:cyan

"""

# Necessary for autodoc_type_aliases to recognize the type aliases used in the signatures
# of methods defined in this module.
from __future__ import annotations

import inspect

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
from pyvista.core.utilities.misc import has_module

from . import _vtk

if TYPE_CHECKING:  # pragma: no cover
    from ._typing import ColorLike

IPYGANY_MAP = {
    'reds': 'Reds',
    'spectral': 'Spectral',
}

# Following colors are copied from matplotlib.colors, synonyms (colors with a
# different name but same hex value) are removed and put in the `color_synonyms`
# dictionary. An extra `paraview_background` color is added.
hexcolors = {
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
    'brown': '#654321',
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
    'paraview_background': '#52576e',
    'peachpuff': '#FFDAB9',
    'peru': '#CD853F',
    'pink': '#FFC0CB',
    'plum': '#DDA0DD',
    'powderblue': '#B0E0E6',
    'purple': '#800080',
    'raw_sienna': '#965434',
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

color_names = {h.lower(): n for n, h in hexcolors.items()}

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

color_synonyms = {
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
    "spectrum": {
        "id": _vtk.vtkColorSeries.SPECTRUM,
        "descr": "black, red, blue, green, purple, orange, brown",
    },
    "warm": {"id": _vtk.vtkColorSeries.WARM, "descr": "dark red → yellow"},
    "cool": {"id": _vtk.vtkColorSeries.COOL, "descr": "green → blue → purple"},
    "blues": {"id": _vtk.vtkColorSeries.BLUES, "descr": "Different shades of blue"},
    "wild_flower": {"id": _vtk.vtkColorSeries.WILD_FLOWER, "descr": "blue → purple → pink"},
    "citrus": {"id": _vtk.vtkColorSeries.CITRUS, "descr": "green → yellow → orange"},
    "div_purple_orange11": {
        "id": _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_11,
        "descr": "dark brown → white → dark purple",
    },
    "div_purple_orange10": {
        "id": _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_10,
        "descr": "dark brown → white → dark purple",
    },
    "div_purple_orange9": {
        "id": _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_9,
        "descr": "brown → white → purple",
    },
    "div_purple_orange8": {
        "id": _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_8,
        "descr": "brown → white → purple",
    },
    "div_purple_orange7": {
        "id": _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_7,
        "descr": "brown → white → purple",
    },
    "div_purple_orange6": {
        "id": _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_6,
        "descr": "brown → white → purple",
    },
    "div_purple_orange5": {
        "id": _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_5,
        "descr": "orange → white → purple",
    },
    "div_purple_orange4": {
        "id": _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_4,
        "descr": "orange → white → purple",
    },
    "div_purple_orange3": {
        "id": _vtk.vtkColorSeries.BREWER_DIVERGING_PURPLE_ORANGE_3,
        "descr": "orange → white → purple",
    },
    "div_spectral11": {
        "id": _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_11,
        "descr": "dark red → light yellow → dark blue",
    },
    "div_spectral10": {
        "id": _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_10,
        "descr": "dark red → light yellow → dark blue",
    },
    "div_spectral9": {
        "id": _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_9,
        "descr": "red → light yellow → blue",
    },
    "div_spectral8": {
        "id": _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_8,
        "descr": "red → light yellow → blue",
    },
    "div_spectral7": {
        "id": _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_7,
        "descr": "red → light yellow → blue",
    },
    "div_spectral6": {
        "id": _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_6,
        "descr": "red → light yellow → blue",
    },
    "div_spectral5": {
        "id": _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_5,
        "descr": "red → light yellow → blue",
    },
    "div_spectral4": {
        "id": _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_4,
        "descr": "red → light yellow → blue",
    },
    "div_spectral3": {
        "id": _vtk.vtkColorSeries.BREWER_DIVERGING_SPECTRAL_3,
        "descr": "orange → light yellow → green",
    },
    "div_brown_blue_green11": {
        "id": _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_11,
        "descr": "dark brown → white → dark blue-green",
    },
    "div_brown_blue_green10": {
        "id": _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_10,
        "descr": "dark brown → white → dark blue-green",
    },
    "div_brown_blue_green9": {
        "id": _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_9,
        "descr": "brown → white → blue-green",
    },
    "div_brown_blue_green8": {
        "id": _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_8,
        "descr": "brown → white → blue-green",
    },
    "div_brown_blue_green7": {
        "id": _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_7,
        "descr": "brown → white → blue-green",
    },
    "div_brown_blue_green6": {
        "id": _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_6,
        "descr": "brown → white → blue-green",
    },
    "div_brown_blue_green5": {
        "id": _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_5,
        "descr": "brown → white → blue-green",
    },
    "div_brown_blue_green4": {
        "id": _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_4,
        "descr": "brown → white → blue-green",
    },
    "div_brown_blue_green3": {
        "id": _vtk.vtkColorSeries.BREWER_DIVERGING_BROWN_BLUE_GREEN_3,
        "descr": "brown → white → blue-green",
    },
    "seq_blue_green9": {
        "id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_9,
        "descr": "light blue → dark green",
    },
    "seq_blue_green8": {
        "id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_8,
        "descr": "light blue → dark green",
    },
    "seq_blue_green7": {
        "id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_7,
        "descr": "light blue → dark green",
    },
    "seq_blue_green6": {
        "id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_6,
        "descr": "light blue → green",
    },
    "seq_blue_green5": {
        "id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_5,
        "descr": "light blue → green",
    },
    "seq_blue_green4": {
        "id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_4,
        "descr": "light blue → green",
    },
    "seq_blue_green3": {
        "id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_GREEN_3,
        "descr": "light blue → green",
    },
    "seq_yellow_orange_brown9": {
        "id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_YELLOW_ORANGE_BROWN_9,
        "descr": "light yellow → orange → dark brown",
    },
    "seq_yellow_orange_brown8": {
        "id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_YELLOW_ORANGE_BROWN_8,
        "descr": "light yellow → orange → brown",
    },
    "seq_yellow_orange_brown7": {
        "id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_YELLOW_ORANGE_BROWN_7,
        "descr": "light yellow → orange → brown",
    },
    "seq_yellow_orange_brown6": {
        "id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_YELLOW_ORANGE_BROWN_6,
        "descr": "light yellow → orange → brown",
    },
    "seq_yellow_orange_brown5": {
        "id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_YELLOW_ORANGE_BROWN_5,
        "descr": "light yellow → orange → brown",
    },
    "seq_yellow_orange_brown4": {
        "id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_YELLOW_ORANGE_BROWN_4,
        "descr": "light yellow → orange",
    },
    "seq_yellow_orange_brown3": {
        "id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_YELLOW_ORANGE_BROWN_3,
        "descr": "light yellow → orange",
    },
    "seq_blue_purple9": {
        "id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_PURPLE_9,
        "descr": "light blue → dark purple",
    },
    "seq_blue_purple8": {
        "id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_PURPLE_8,
        "descr": "light blue → purple",
    },
    "seq_blue_purple7": {
        "id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_PURPLE_7,
        "descr": "light blue → purple",
    },
    "seq_blue_purple6": {
        "id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_PURPLE_6,
        "descr": "light blue → purple",
    },
    "seq_blue_purple5": {
        "id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_PURPLE_5,
        "descr": "light blue → purple",
    },
    "seq_blue_purple4": {
        "id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_PURPLE_4,
        "descr": "light blue → purple",
    },
    "seq_blue_purple3": {
        "id": _vtk.vtkColorSeries.BREWER_SEQUENTIAL_BLUE_PURPLE_3,
        "descr": "light blue → purple",
    },
    "qual_accent": {
        "id": _vtk.vtkColorSeries.BREWER_QUALITATIVE_ACCENT,
        "descr": "pastel green, pastel purple, pastel orange, pastel yellow, blue, pink, brown, gray",
    },
    "qual_dark2": {
        "id": _vtk.vtkColorSeries.BREWER_QUALITATIVE_DARK2,
        "descr": "darker shade of qual_set2",
    },
    "qual_set3": {
        "id": _vtk.vtkColorSeries.BREWER_QUALITATIVE_SET3,
        "descr": "pastel colors: blue green, light yellow, dark purple, red, blue, orange, green, pink, gray, purple, light green, yellow",
    },
    "qual_set2": {
        "id": _vtk.vtkColorSeries.BREWER_QUALITATIVE_SET2,
        "descr": "blue green, orange, purple, pink, green, yellow, brown, gray",
    },
    "qual_set1": {
        "id": _vtk.vtkColorSeries.BREWER_QUALITATIVE_SET1,
        "descr": "red, blue, green, purple, orange, yellow, brown, pink, gray",
    },
    "qual_pastel2": {
        "id": _vtk.vtkColorSeries.BREWER_QUALITATIVE_PASTEL2,
        "descr": "pastel shade of qual_set2",
    },
    "qual_pastel1": {
        "id": _vtk.vtkColorSeries.BREWER_QUALITATIVE_PASTEL1,
        "descr": "pastel shade of qual_set1",
    },
    "qual_paired": {
        "id": _vtk.vtkColorSeries.BREWER_QUALITATIVE_PAIRED,
        "descr": "light blue, blue, light green, green, light red, red, light orange, orange, light purple, purple, light yellow",
    },
    "custom": {"id": _vtk.vtkColorSeries.CUSTOM, "descr": None},
}

SCHEME_NAMES = {
    scheme_info["id"]: scheme_name for scheme_name, scheme_info in COLOR_SCHEMES.items()
}


class Color:
    """Helper class to convert between different color representations used in the pyvista library.

    Many pyvista methods accept :data:`ColorLike` parameters. This helper class
    is used to convert such parameters to the necessary format, used by
    underlying (VTK) methods. Any color name (``str``), hex string (``str``)
    or RGB(A) sequence (``tuple``, ``list`` or ``numpy.ndarray`` of ``int``
    or ``float``) is considered a :data:`ColorLike` parameter and can be converted
    by this class.
    See :attr:`Color.name` for a list of supported color names.

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

    Notes
    -----
    The internally used representation is an integer RGBA sequence (values
    between 0 and 255). This might however change in future releases.

    .. raw:: html

       <details><summary>Refer to the table below for a list of supported colors.</summary>

    .. include:: ../color_table/color_table.rst

    .. raw:: html

       </details>

    Examples
    --------
    Create a transparent green color using a color name, float RGBA sequence,
    integer RGBA sequence and RGBA hexadecimal string.

    >>> import pyvista as pv
    >>> pv.Color("green", opacity=0.5)
    Color(name='green', hex='#00800080', opacity=128)
    >>> pv.Color([0.0, 0.5, 0.0, 0.5])
    Color(name='green', hex='#00800080', opacity=128)
    >>> pv.Color([0, 128, 0, 128])
    Color(name='green', hex='#00800080', opacity=128)
    >>> pv.Color("#00800080")
    Color(name='green', hex='#00800080', opacity=128)

    """

    # Supported names for each color channel.
    CHANNEL_NAMES = (
        {'red', 'r'},  # 0
        {'green', 'g'},  # 1
        {'blue', 'b'},  # 2
        {'alpha', 'a', 'opacity'},  # 3
    )

    def __init__(
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
            else:
                raise ValueError(f"Unsupported color type: {type(color)}")
            self._name = color_names.get(self.hex_rgb, None)
        except ValueError as e:
            raise ValueError(
                "\n"
                f"\tInvalid color input: ({color})\n"
                "\tMust be a string, rgb(a) sequence, or hex color string.  For example:\n"
                "\t\tcolor='white'\n"
                "\t\tcolor='w'\n"
                "\t\tcolor=[1.0, 1.0, 1.0]\n"
                "\t\tcolor=[255, 255, 255]\n"
                "\t\tcolor='#FFFFFF'",
            ) from e

        # Overwrite opacity if it is provided
        try:
            if opacity is not None:
                self._opacity = self.convert_color_channel(opacity)
        except ValueError as e:
            raise ValueError(
                "\n"
                f"\tInvalid opacity input: ({opacity})"
                "\tMust be an integer, float or string.  For example:\n"
                "\t\topacity='1.0'\n"
                "\t\topacity='255'\n"
                "\t\topacity='#FF'",
            ) from e

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
        if h.startswith('0x'):
            h = h[2:]
        return h

    @staticmethod
    def convert_color_channel(
        val: float | np.floating[Any] | str,
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
        if isinstance(val, str):
            # From hexadecimal value
            val = int(Color.strip_hex_prefix(val), 16)
        elif np.issubdtype(np.asarray(val).dtype, np.floating) and np.ndim(val) == 0:
            # From float
            val = int(round(255 * val))
        if (
            np.issubdtype(np.asarray(val).dtype, np.integer)
            and np.size(val) == 1
            and 0 <= val <= 255
        ):
            # From integer
            return int(val)
        else:
            raise ValueError(f"Unsupported color channel value provided: {val}")

    def _from_rgba(self, rgba):
        """Construct color from an RGB(A) sequence."""
        arg = rgba
        if len(rgba) == 3:
            # Keep using current opacity if it is not provided.
            rgba = [*rgba, self._opacity]
        try:
            if len(rgba) != 4:
                raise ValueError("Invalid length for RGBA sequence.")
            self._red, self._green, self._blue, self._opacity = (
                self.convert_color_channel(c) for c in rgba
            )
        except ValueError:
            raise ValueError(f"Invalid RGB(A) sequence: {arg}") from None

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
            self._from_rgba([self.convert_color_channel(h[i : i + 2]) for i in range(0, len(h), 2)])
        except ValueError:
            raise ValueError(f"Invalid hex string: {arg}") from None

    def _from_str(self, n: str):
        """Construct color from a name or hex string."""
        arg = n
        n = n.lower()
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
                raise ValueError(f"Invalid color name or hex string: {arg}") from None

    @property
    def int_rgba(self) -> tuple[int, int, int, int]:  # numpydoc ignore=RT01
        """Get the color value as an RGBA integer tuple.

        Examples
        --------
        Create a blue color with half opacity.

        >>> import pyvista as pv
        >>> c = pv.Color("blue", opacity=128)
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
        >>> c = pv.Color("blue", opacity=128)
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
        >>> c = pv.Color("blue", opacity=0.6)
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
        return self._red / 255.0, self._green / 255.0, self._blue / 255.0, self._opacity / 255.0

    @property
    def float_rgb(self) -> tuple[float, float, float]:  # numpydoc ignore=RT01
        """Get the color value as an RGB float tuple.

        Examples
        --------
        Create a blue color with custom opacity.

        >>> import pyvista as pv
        >>> c = pv.Color("blue", default_opacity=0.6)
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
    def hex_rgba(self) -> str:  # numpydoc ignore=RT01
        """Get the color value as an RGBA hexadecimal value.

        Examples
        --------
        Create a blue color with half opacity.

        >>> import pyvista as pv
        >>> c = pv.Color("blue", default_opacity="#80")
        >>> c
        Color(name='blue', hex='#0000ff80', opacity=128)
        >>> c.hex_rgba
        '#0000ff80'

        Create a transparent red color using an RGBA hexadecimal value.

        >>> c = pv.Color("0xff000040")
        >>> c
        Color(name='red', hex='#ff000040', opacity=64)
        >>> c.hex_rgba
        '#ff000040'

        """
        return '#' + ''.join(
            f"{c:0>2x}" for c in (self._red, self._green, self._blue, self._opacity)
        )

    @property
    def hex_rgb(self) -> str:  # numpydoc ignore=RT01
        """Get the color value as an RGB hexadecimal value.

        Examples
        --------
        Create a blue color with half opacity.

        >>> import pyvista as pv
        >>> c = pv.Color("blue", default_opacity="#80")
        >>> c
        Color(name='blue', hex='#0000ff80', opacity=128)
        >>> c.hex_rgb
        '#0000ff'

        Create a red color using an RGB hexadecimal value.

        >>> c = pv.Color("0xff0000")
        >>> c
        Color(name='red', hex='#ff0000ff', opacity=255)
        >>> c.hex_rgb
        '#ff0000'

        """
        return self.hex_rgba[:-2]

    @property
    def name(self) -> str | None:  # numpydoc ignore=RT01
        """Get the color name.

        Returns
        -------
        str | None
            The color name, in case this color has a name; otherwise ``None``.

        Notes
        -----
        Refer to the table below for a list of supported colors.

        .. include:: ../colors.rst

        Examples
        --------
        Create a blue color with half opacity.

        >>> import pyvista as pv
        >>> c = pv.Color("blue", default_opacity=0.5)
        >>> c
        Color(name='blue', hex='#0000ff80', opacity=128)

        """
        return self._name

    @property
    def vtk_c3ub(self) -> _vtk.vtkColor3ub:  # numpydoc ignore=RT01
        """Get the color value as a VTK Color3ub instance.

        Examples
        --------
        Create a blue color with half opacity.

        >>> import pyvista as pv
        >>> c = pv.Color("blue", default_opacity=0.5)
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
    def from_dict(cls, dict_):
        """Construct from dictionary for JSON deserialization."""
        return Color(dict_)

    def to_dict(self):
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
            raise TypeError("Invalid index specified, only strings and integers are supported.")
        if isinstance(item, str):
            for i, cnames in enumerate(self.CHANNEL_NAMES):
                if item in cnames:
                    item = i
                    break
            else:
                raise ValueError(f"Invalid string index {item!r}.")
        return self.float_rgba[item]

    def __iter__(self):
        """Support iteration over the float RGBA representation for backward compatibility."""
        return iter(self.float_rgba)

    def __repr__(self):  # pragma: no cover
        """Human readable representation."""
        kwargs = f"hex={self.hex_rgba!r}, opacity={self.opacity}"
        if self._name is not None:
            kwargs = f"name={self._name!r}, " + kwargs
        return f"Color({kwargs})"


PARAVIEW_BACKGROUND = Color('paraview').float_rgb  # [82, 87, 110] / 255


def get_cmap_safe(cmap):
    """
    Fetch a colormap by name from matplotlib, colorcet, or cmocean.

    Parameters
    ----------
    cmap : str or list of str
        Name of the colormap to fetch. If the input is a list of strings,
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
    if isinstance(cmap, str):
        # check if this colormap has been mapped between ipygany
        if cmap in IPYGANY_MAP:
            cmap = IPYGANY_MAP[cmap]

        # Try colorcet first
        if has_module('colorcet'):
            import colorcet

            try:
                return colorcet.cm[cmap]
            except KeyError:
                pass

        # Try cmocean second
        if has_module('cmocean'):
            import cmocean

            try:
                return getattr(cmocean.cm, cmap)
            except AttributeError:
                pass

        if not isinstance(cmap, colors.Colormap):
            if inspect.ismodule(colormaps):  # pragma: no cover
                # Backwards compatibility with matplotlib<3.5.0
                if not hasattr(colormaps, cmap):
                    raise ValueError(f'Invalid colormap "{cmap}"')
                cmap = getattr(colormaps, cmap)
            else:
                try:
                    cmap = colormaps[cmap]
                except KeyError:
                    raise ValueError(f'Invalid colormap "{cmap}"') from None

    elif isinstance(cmap, list):
        for item in cmap:
            if not isinstance(item, str):
                raise TypeError('When inputting a list as a cmap, each item should be a string.')

        cmap = ListedColormap(cmap)

    return cmap


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
    """
    Return the color cycler of the current matplotlib theme.

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
    scheme : str, int, or _vtk.vtkColorSeries
        Color scheme to be converted. If a string, it should correspond to a
        valid color scheme name (e.g., 'viridis'). If an integer, it should
        correspond to a valid color scheme ID. If an instance of
        `_vtk.vtkColorSeries`, it should be a valid color series.

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
            series.SetColorScheme(COLOR_SCHEMES.get(scheme.lower())["id"])
        elif isinstance(scheme, int):
            series.SetColorScheme(scheme)
        else:
            raise ValueError(f'Color scheme not understood: {scheme}')
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
            - ``'all'``: Cycle through all available colors in ``pyvista.plotting.colors.hexcolors``
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
        return None
    elif isinstance(color_cycler, str):
        if color_cycler == 'default':
            return get_default_cycler()
        elif color_cycler == 'matplotlib':
            return get_matplotlib_theme_cycler()
        elif color_cycler == 'all':
            return get_hexcolors_cycler()
        elif color_cycler in COLOR_SCHEMES:
            return color_scheme_to_cycler(color_cycler)
        else:
            raise ValueError(f'color cycler of name `{color_cycler}` not found.')
    elif isinstance(color_cycler, (tuple, list)):
        return cycler('color', color_cycler)
    elif isinstance(color_cycler, Cycler):
        return color_cycler
    else:
        raise TypeError(f'color cycler of type {type(color_cycler)} not supported.')
