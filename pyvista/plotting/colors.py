"""Color module supporting plotting module.

Used code from matplotlib.colors.  Thanks for your work!


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

from __future__ import annotations  # Necessary for autodoc_type_aliases to recognize the 'color_like' alias
import numpy as np
from typing import Optional, Sequence, Tuple, Union
import pyvista
from pyvista import _vtk

color3i = Union[Tuple[int, int, int], Sequence[int], np.ndarray]
color4i = Union[Tuple[int, int, int, int], Sequence[int], np.ndarray]
color3f = Union[Tuple[float, float, float], Sequence[float], np.ndarray]
color4f = Union[Tuple[float, float, float, float], Sequence[float], np.ndarray]
color_like = Union[
    Tuple[int, int, int], Tuple[int, int, int, int],
    Tuple[float, float, float], Tuple[float, float, float, float],
    Sequence[int], Sequence[float], np.ndarray, str, "Color",
    _vtk.vtkColor3ub
]
"""Any object convertible to a :class:`Color`."""

IPYGANY_MAP = {
    'reds': 'Reds',
    'spectral': 'Spectral',
}

# shamelessly copied from matplotlib.colors
hexcolors = {
    'aliceblue':            '#F0F8FF',
    'antiquewhite':         '#FAEBD7',
    'aqua':                 '#00FFFF',
    'aquamarine':           '#7FFFD4',
    'azure':                '#F0FFFF',
    'beige':                '#F5F5DC',
    'bisque':               '#FFE4C4',
    'black':                '#000000',
    'blanchedalmond':       '#FFEBCD',
    'blue':                 '#0000FF',
    'blueviolet':           '#8A2BE2',
    'brown':                '#654321',
    'burlywood':            '#DEB887',
    'cadetblue':            '#5F9EA0',
    'chartreuse':           '#7FFF00',
    'chocolate':            '#D2691E',
    'coral':                '#FF7F50',
    'cornflowerblue':       '#6495ED',
    'cornsilk':             '#FFF8DC',
    'crimson':              '#DC143C',
    'cyan':                 '#00FFFF',
    'darkblue':             '#00008B',
    'darkcyan':             '#008B8B',
    'darkgoldenrod':        '#B8860B',
    'darkgray':             '#A9A9A9',
    'darkgreen':            '#006400',
    'darkgrey':             '#A9A9A9',
    'darkkhaki':            '#BDB76B',
    'darkmagenta':          '#8B008B',
    'darkolivegreen':       '#556B2F',
    'darkorange':           '#FF8C00',
    'darkorchid':           '#9932CC',
    'darkred':              '#8B0000',
    'darksalmon':           '#E9967A',
    'darkseagreen':         '#8FBC8F',
    'darkslateblue':        '#483D8B',
    'darkslategray':        '#2F4F4F',
    'darkslategrey':        '#2F4F4F',
    'darkturquoise':        '#00CED1',
    'darkviolet':           '#9400D3',
    'deeppink':             '#FF1493',
    'deepskyblue':          '#00BFFF',
    'dimgray':              '#696969',
    'dimgrey':              '#696969',
    'dodgerblue':           '#1E90FF',
    'firebrick':            '#B22222',
    'floralwhite':          '#FFFAF0',
    'forestgreen':          '#228B22',
    'fuchsia':              '#FF00FF',
    'gainsboro':            '#DCDCDC',
    'ghostwhite':           '#F8F8FF',
    'gold':                 '#FFD700',
    'goldenrod':            '#DAA520',
    'gray':                 '#808080',
    'green':                '#008000',
    'greenyellow':          '#ADFF2F',
    'grey':                 '#808080',
    'honeydew':             '#F0FFF0',
    'hotpink':              '#FF69B4',
    'indianred':            '#CD5C5C',
    'indigo':               '#4B0082',
    'ivory':                '#FFFFF0',
    'khaki':                '#F0E68C',
    'lavender':             '#E6E6FA',
    'lavenderblush':        '#FFF0F5',
    'lawngreen':            '#7CFC00',
    'lemonchiffon':         '#FFFACD',
    'lightblue':            '#ADD8E6',
    'lightcoral':           '#F08080',
    'lightcyan':            '#E0FFFF',
    'lightgoldenrodyellow': '#FAFAD2',
    'lightgray':            '#D3D3D3',
    'lightgreen':           '#90EE90',
    'lightgrey':            '#D3D3D3',
    'lightpink':            '#FFB6C1',
    'lightsalmon':          '#FFA07A',
    'lightseagreen':        '#20B2AA',
    'lightskyblue':         '#87CEFA',
    'lightslategray':       '#778899',
    'lightslategrey':       '#778899',
    'lightsteelblue':       '#B0C4DE',
    'lightyellow':          '#FFFFE0',
    'lime':                 '#00FF00',
    'limegreen':            '#32CD32',
    'linen':                '#FAF0E6',
    'magenta':              '#FF00FF',
    'maroon':               '#800000',
    'mediumaquamarine':     '#66CDAA',
    'mediumblue':           '#0000CD',
    'mediumorchid':         '#BA55D3',
    'mediumpurple':         '#9370DB',
    'mediumseagreen':       '#3CB371',
    'mediumslateblue':      '#7B68EE',
    'mediumspringgreen':    '#00FA9A',
    'mediumturquoise':      '#48D1CC',
    'mediumvioletred':      '#C71585',
    'midnightblue':         '#191970',
    'mintcream':            '#F5FFFA',
    'mistyrose':            '#FFE4E1',
    'moccasin':             '#FFE4B5',
    'navajowhite':          '#FFDEAD',
    'navy':                 '#000080',
    'oldlace':              '#FDF5E6',
    'olive':                '#808000',
    'olivedrab':            '#6B8E23',
    'orange':               '#FFA500',
    'orangered':            '#FF4500',
    'orchid':               '#DA70D6',
    'palegoldenrod':        '#EEE8AA',
    'palegreen':            '#98FB98',
    'paleturquoise':        '#AFEEEE',
    'palevioletred':        '#DB7093',
    'papayawhip':           '#FFEFD5',
    'peachpuff':            '#FFDAB9',
    'peru':                 '#CD853F',
    'pink':                 '#FFC0CB',
    'plum':                 '#DDA0DD',
    'powderblue':           '#B0E0E6',
    'purple':               '#800080',
    'raw_sienna':           '#965434',
    'rebeccapurple':        '#663399',
    'red':                  '#FF0000',
    'rosybrown':            '#BC8F8F',
    'royalblue':            '#4169E1',
    'saddlebrown':          '#8B4513',
    'salmon':               '#FA8072',
    'sandybrown':           '#F4A460',
    'seagreen':             '#2E8B57',
    'seashell':             '#FFF5EE',
    'sienna':               '#A0522D',
    'silver':               '#C0C0C0',
    'skyblue':              '#87CEEB',
    'slateblue':            '#6A5ACD',
    'slategray':            '#708090',
    'slategrey':            '#708090',
    'snow':                 '#FFFAFA',
    'springgreen':          '#00FF7F',
    'steelblue':            '#4682B4',
    'tan':                  '#D2B48C',
    'teal':                 '#008080',
    'thistle':              '#D8BFD8',
    'tomato':               '#FF6347',
    'turquoise':            '#40E0D0',
    'violet':               '#EE82EE',
    'wheat':                '#F5DEB3',
    'white':                '#FFFFFF',
    'whitesmoke':           '#F5F5F5',
    'yellow':               '#FFFF00',
    'yellowgreen':          '#9ACD32',
    'tab:blue':             '#1f77b4',
    'tab:orange':           '#ff7f0e',
    'tab:green':            '#2ca02c',
    'tab:red':              '#d62728',
    'tab:purple':           '#9467bd',
    'tab:brown':            '#8c564b',
    'tab:pink':             '#e377c2',
    'tab:gray':             '#7f7f7f',
    'tab:olive':            '#bcbd22',
    'tab:cyan':             '#17becf',
}

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


PARAVIEW_BACKGROUND = [82/255., 87/255., 110/255.]


class Color:
    """Helper class to convert between different color representations used in the pyvista library.

    Many pyvista methods accept :attr:`color_like` parameters. This helper class
    is used to convert such parameters to the necessary format, used by
    underlying (VTK) methods. Any color name (``str``), hex string (``str``)
    or RGB(A) sequence (``tuple``, ``list`` or ``numpy.ndarray`` of ``int``
    or ``float``) is considered a :attr:`color_like` parameter and can be converted
    by this class.
    See :attr:`Color.name` for a list of supported color names.

    Parameters
    ----------
    color : color_like, optional
        Either a string, RGB sequence, RGBA sequence, or hex color string.
        RGB(A) sequences should either be provided as floats between 0 and 1
        or as ints between 0 and 255. If no opacity is provided, the
        `default_opacity` will be used. If `color` is ``None``, the
        `default_color` is used instead.
        The following examples all denote the color 'white':

        * ``'white'``
        * ``'w'``
        * ``[1.0, 1.0, 1.0]``
        * ``[255, 255, 255, 255]``
        * ``'#FFFFFF'``

    default_opacity : int, float or str, optional
        Default opacity of the represented color. Used when `color`
        does not specify an opacity. Allowed opacities are floats between 0
        and 1, ints between 0 and 255 or hexadecimal strings of length 2.
        The following examples all denote a fully opaque color:

        * ``1.0``
        * ``255``
        * ``'#ff'``

    default_color : color_like, optional
        Default color to use when `color` is None. If this value is
        ``None``, then defaults to the global theme color. Format is
        identical to `color`.

    Notes
    -----
    The internally used representation is an integer RGBA sequence (values
    between 0 and 255). This might however change in future releases.

    """

    def __init__(self, color: Optional[color_like] = None, default_opacity: Union[int, float, str] = 255,
                 default_color: Optional[color_like] = None):
        """Initialize new instance."""
        self._red, self._green, self._blue, self._opacity = 0, 0, 0, 0
        self._default_opacity = self.convert_color_channel(default_opacity)
        self._name = None

        # Use default color if no color is provided
        if color is None:
            if default_color is None:
                color = pyvista.global_theme.color
            else:
                color = default_color

        try:
            if isinstance(color, Color):
                # Create copy of color instance (but keep own defaults)
                self._red, self._green, self._blue, self._opacity = color.i_rgba
            elif isinstance(color, str):
                # From named color or hex string
                self.name = color
            elif isinstance(color, (list, tuple, np.ndarray)):
                # From RGB(A) sequence
                if np.issubdtype(np.asarray(color).dtype, np.floating):
                    self.f_rgba = color  # type: ignore
                else:
                    self.i_rgba = color  # type: ignore
            elif isinstance(color, _vtk.vtkColor3ub):
                # From vtkColor3ub instance
                self.vtk_c3ub = color
            else:
                raise ValueError(f"Unsupported color type: {type(color)}")
        except ValueError as e:
            raise ValueError("\n"
                             f"\tInvalid color input: ({color})\n"
                             "\tMust be a string, rgb(a) sequence, or hex color string.  For example:\n"
                             "\t\tcolor='white'\n"
                             "\t\tcolor='w'\n"
                             "\t\tcolor=[1, 1, 1]\n"
                             "\t\tcolor='#FFFFFF'") from e

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
    def convert_color_channel(val: Union[int, np.integer, float, np.floating, str]) -> int:
        """Convert the given color channel value to the integer representation (values between ``0`` and ``255``).

        Parameters
        ----------
        val : int, float or str
            Color channel value to convert. Supported input values are a hex string of length 2 (``'00'`` to ``'ff'``),
            a float (``0.0`` to ``1.0``) or an integer (``0`` to ``255``).

        Returns
        -------
        int
            Color channel value in the integer representation.

        """
        if isinstance(val, str):
            # From hexadecimal value
            val = int(Color.strip_hex_prefix(val), 16)
        elif np.issubdtype(np.asarray(val).dtype, np.floating) and np.ndim(val) == 0:
            # From float
            val = int(255 * val + 0.5)
        if np.issubdtype(np.asarray(val).dtype, np.integer) and np.ndim(val) == 0 and 0 <= val <= 255:
            # From integer
            return int(val)
        else:
            raise ValueError(f"Unsupported color channel value provided: {val}")

    @property
    def i_rgba(self) -> Tuple[int, int, int, int]:
        """Get or set the color value as an RGB(A) integer sequence.

        Examples
        --------
        Create a blue color with half opacity.

        >>> import pyvista
        >>> c = pyvista.Color("blue", default_opacity=128)
        >>> c
        Color(name='blue', hex='#0000ff80')
        >>> c.i_rgba
        (0, 0, 255, 128)

        Modify the RGBA values of the color using an integer sequence.

        >>> c.i_rgba = [255, 0, 0, 64]
        >>> c
        Color(hex='#ff000040')

        Modify the RGB values of the color using an integer sequence.
        This uses the ``default_opacity``.

        >>> c.i_rgba = [0, 255, 0]
        >>> c
        Color(hex='#00ff0080')

        """
        return self._red, self._green, self._blue, self._opacity

    @i_rgba.setter
    def i_rgba(self, rgba: Union[color3i, color4i]):
        if len(rgba) == 3:
            rgba = [*rgba, self._default_opacity]
        try:
            if len(rgba) != 4:
                raise ValueError("Invalid length for RGBA sequence.")
            self._red, self._green, self._blue, self._opacity = [self.convert_color_channel(c) for c in rgba]
            self._name = None
        except ValueError as e:
            raise ValueError(f"Invalid RGB(A) sequence: {rgba}") from e

    @property
    def i_rgb(self) -> Tuple[int, int, int]:
        """Get or set the color value as an RGB integer sequence.

        Examples
        --------
        Create a blue color with half opacity.

        >>> import pyvista
        >>> c = pyvista.Color("blue", default_opacity=128)
        >>> c
        Color(name='blue', hex='#0000ff80')
        >>> c.i_rgb
        (0, 0, 255)

        Modify the RGB values of the color using an integer sequence.
        This uses the ``default_opacity``.

        >>> c.i_rgb = [0, 255, 0]
        >>> c
        Color(hex='#00ff0080')

        """
        return self.i_rgba[:3]

    @i_rgb.setter
    def i_rgb(self, rgb: color3i):
        self.i_rgba = rgb  # type: ignore

    @property
    def f_rgba(self) -> Tuple[float, float, float, float]:
        """Get or set the color value as an RGB(A) float sequence.

        Examples
        --------
        Create a blue color with half opacity.

        >>> import pyvista
        >>> c = pyvista.Color("blue", default_opacity=0.6)
        >>> c
        Color(name='blue', hex='#0000ff99')
        >>> c.f_rgba
        (0.0, 0.0, 1.0, 0.6)

        Modify the RGBA values of the color using a float sequence.

        >>> c.f_rgba = [1.0, 0.0, 0.0, 0.4]
        >>> c
        Color(hex='#ff000066')

        Modify the RGB values of the color using a float sequence.
        This uses the ``default_opacity``.

        >>> c.f_rgba = [0.0, 1.0, 0.0]
        >>> c
        Color(hex='#00ff0099')

        """
        return self._red / 255.0, self._green / 255.0, self._blue / 255.0, self._opacity / 255.0

    @f_rgba.setter
    def f_rgba(self, rgba: Union[color3f, color4f]):
        try:
            self.i_rgba = [self.convert_color_channel(c) for c in rgba]  # type: ignore
        except ValueError as e:
            raise ValueError(f"Invalid RGB(A) sequence: {rgba}") from e

    @property
    def f_rgb(self) -> Tuple[float, float, float]:
        """Get or set the color value as an RGB float sequence.

        Examples
        --------
        Create a blue color with half opacity.

        >>> import pyvista
        >>> c = pyvista.Color("blue", default_opacity=0.5)
        >>> c
        Color(name='blue', hex='#0000ff80')
        >>> c.f_rgb
        (0.0, 0.0, 1.0)

        Modify the RGB values of the color using a float sequence.
        This uses the ``default_opacity``.

        >>> c.f_rgb = [0.0, 1.0, 0.0]
        >>> c
        Color(hex='#00ff0080')

        """
        return self.f_rgba[:3]

    @f_rgb.setter
    def f_rgb(self, rgb: color3f):
        self.f_rgba = rgb  # type: ignore

    @property
    def hex(self) -> str:
        """Get or set the color value as an RGB(A) hexadecimal value.

        Examples
        --------
        Create a blue color with half opacity.

        >>> import pyvista
        >>> c = pyvista.Color("blue", default_opacity="#80")
        >>> c
        Color(name='blue', hex='#0000ff80')

        Modify the RGBA values of the color using a hexadecimal value.

        >>> c.hex = "0xff000040"
        >>> c
        Color(hex='#ff000040')

        Modify the RGB values of the color using a hexadecimal value.
        This uses the ``default_opacity``.

        >>> c.hex = "#00ff00"
        >>> c
        Color(hex='#00ff0080')

        """
        return '#' + ''.join(f"{c:0>2x}" for c in (self._red, self._green, self._blue, self._opacity))

    @hex.setter
    def hex(self, h: str):
        h = self.strip_hex_prefix(h)
        try:
            self.i_rgba = [self.convert_color_channel(h[i:i+2]) for i in range(0, len(h), 2)]  # type: ignore
        except ValueError as e:
            raise ValueError(f"Invalid hex string: {h}") from e

    @property
    def name(self) -> Optional[str]:
        """Get or set the color value by name or hexadecimal value.

        Notes
        -----
        Refer to the table below for a list of supported colors.

        .. include:: ../colors.rst

        Examples
        --------
        Create a blue color with half opacity.

        >>> import pyvista
        >>> c = pyvista.Color("blue", default_opacity=0.5)
        >>> c
        Color(name='blue', hex='#0000ff80')

        Modify the color by name.

        >>> c.name = "red"
        >>> c
        Color(name='red', hex='#ff000080')

        """
        return self._name

    @name.setter
    def name(self, n: str):
        n = n.lower()
        if len(n) == 1:
            # Single character
            # Convert from single character to full hex
            if n not in color_char_to_word:
                raise ValueError('Single character string must be one of the following:'
                                 f'\n{str(color_char_to_word.keys())}')
            n = color_char_to_word[n]
            self.hex = hexcolors[n]
            self._name = n
        elif n in hexcolors:
            # Color name
            self.hex = hexcolors[n]
            self._name = n
        elif n in 'paraview' or n in 'pv':
            # Use the default ParaView background color
            self.f_rgba = PARAVIEW_BACKGROUND  # type: ignore
            self._name = 'paraview'
        else:
            # Otherwise, try conversion to hex
            try:
                self.hex = n
            except ValueError as e:
                raise ValueError(f"Invalid color name or hex string: {n}") from e

    @property
    def vtk_c3ub(self) -> _vtk.vtkColor3ub:
        """Get or set the color value as a VTK Color3ub instance.

        Examples
        --------
        Create a blue color with half opacity.

        >>> import pyvista
        >>> c = pyvista.Color("blue", default_opacity=0.5)
        >>> c
        Color(name='blue', hex='#0000ff80')
        >>> c.vtk_c3ub
        vtkmodules.vtkCommonDataModel.vtkColor3ub([0, 0, 255])

        """
        return _vtk.vtkColor3ub(self._red, self._green, self._blue)

    @vtk_c3ub.setter
    def vtk_c3ub(self, c3ub: _vtk.vtkColor3ub):
        self.i_rgba = c3ub
        self._name = None

    def __eq__(self, other):
        """Equality comparison."""
        try:
            return self.i_rgba == Color(other).i_rgba
        except ValueError:
            return NotImplemented

    def __repr__(self):
        """Human readable representation."""
        kwargs = f"hex='{self.hex}'"
        if self._name is not None:
            kwargs = f"name='{self._name}', " + kwargs
        return f"Color({kwargs})"


def hex_to_rgb(h):
    """Return 0 to 1 rgb from a hex list or tuple."""
    return Color(h).f_rgb


def string_to_rgb(string):
    """Convert a literal color string (i.e. white) to a color rgb.

    Also accepts hex strings or single characters from the following list.

        b: blue
        g: green
        r: red
        c: cyan
        m: magenta
        y: yellow
        k: black
        w: white

    """
    return Color(string).f_rgb


def get_cmap_safe(cmap):
    """Fetch a colormap by name from matplotlib, colorcet, or cmocean."""
    try:
        from matplotlib.cm import get_cmap
    except ImportError:
        raise ImportError('cmap requires matplotlib')
    if isinstance(cmap, str):
        # check if this colormap has been mapped between ipygany
        if cmap in IPYGANY_MAP:
            cmap = IPYGANY_MAP[cmap]

        # Try colorcet first
        try:
            import colorcet
            cmap = colorcet.cm[cmap]
        except (ImportError, KeyError):
            pass
        else:
            return cmap
        # Try cmocean second
        try:
            import cmocean
            cmap = getattr(cmocean.cm, cmap)
        except (ImportError, AttributeError):
            pass
        else:
            return cmap
        # Else use Matplotlib
        cmap = get_cmap(cmap)
    elif isinstance(cmap, list):
        for item in cmap:
            if not isinstance(item, str):
                raise TypeError('When inputting a list as a cmap, each item should be a string.')
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(cmap)

    return cmap
