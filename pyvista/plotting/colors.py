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

# Necessary for autodoc_type_aliases to recognize the type aliases used in the signatures
# of methods defined in this module.
from __future__ import annotations

from typing import Optional, Tuple, Union
import warnings

import numpy as np

import pyvista
from pyvista import _vtk
from pyvista._typing import color_like
from pyvista.utilities import PyVistaDeprecationWarning
from pyvista.utilities.misc import has_module

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


class Color:
    """Helper class to convert between different color representations used in the pyvista library.

    Many pyvista methods accept :data:`color_like` parameters. This helper class
    is used to convert such parameters to the necessary format, used by
    underlying (VTK) methods. Any color name (``str``), hex string (``str``)
    or RGB(A) sequence (``tuple``, ``list`` or ``numpy.ndarray`` of ``int``
    or ``float``) is considered a :data:`color_like` parameter and can be converted
    by this class.
    See :attr:`Color.name` for a list of supported color names.

    Parameters
    ----------
    color : color_like, optional
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

    opacity : int, float or str, optional
        Opacity of the represented color. Overrides any opacity associated
        with the provided ``color``. Allowed opacities are floats between 0
        and 1, ints between 0 and 255 or hexadecimal strings of length 2
        (plus the length of the optional prefix).
        The following examples all denote a fully opaque color:

        * ``1.0``
        * ``255``
        * ``'#ff'``

    default_color : color_like, optional
        Default color to use when ``color`` is ``None``. If this value is
        ``None``, then defaults to the global theme color. Format is
        identical to ``color``.

    default_opacity : int, float or str, optional
        Default opacity of the represented color. Used when ``color``
        does not specify an opacity and ``opacity`` is ``None``. Format
        is identical to ``opacity``.

    Notes
    -----
    The internally used representation is an integer RGBA sequence (values
    between 0 and 255). This might however change in future releases.

    .. raw:: html

       <details><summary>Refer to the table below for a list of supported colors.</summary>

    .. include:: ../colors.rst

    .. raw:: html

       </details>

    Examples
    --------
    Create a transparent green color using a color name, float RGBA sequence,
    integer RGBA sequence and RGBA hexadecimal string.

    >>> import pyvista
    >>> pyvista.Color("green", opacity=0.5)
    Color(name='green', hex='#00800080')
    >>> pyvista.Color([0.0, 0.5, 0.0, 0.5])
    Color(name='green', hex='#00800080')
    >>> pyvista.Color([0, 128, 0, 128])
    Color(name='green', hex='#00800080')
    >>> pyvista.Color("#00800080")
    Color(name='green', hex='#00800080')

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
        color: Optional[color_like] = None,
        opacity: Optional[Union[int, float, str]] = None,
        default_color: Optional[color_like] = None,
        default_opacity: Union[int, float, str] = 255,
    ):
        """Initialize new instance."""
        self._red, self._green, self._blue, self._opacity = 0, 0, 0, 0
        self._opacity = self.convert_color_channel(default_opacity)
        self._name = None

        # Use default color if no color is provided
        if color is None:
            if default_color is None:
                color = pyvista.global_theme.color
            else:
                color = default_color

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
                "\t\tcolor='#FFFFFF'"
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
                "\t\topacity='#FF'"
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
    def convert_color_channel(val: Union[int, np.integer, float, np.floating, str]) -> int:
        """Convert the given color channel value to the integer representation.

        Parameters
        ----------
        val : int, float or str
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
            self._red, self._green, self._blue, self._opacity = [
                self.convert_color_channel(c) for c in rgba
            ]
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
    def int_rgba(self) -> Tuple[int, int, int, int]:
        """Get the color value as an RGBA integer tuple.

        Examples
        --------
        Create a blue color with half opacity.

        >>> import pyvista
        >>> c = pyvista.Color("blue", opacity=128)
        >>> c
        Color(name='blue', hex='#0000ff80')
        >>> c.int_rgba
        (0, 0, 255, 128)

        Create a transparent red color using an integer RGBA sequence.

        >>> c = pyvista.Color([255, 0, 0, 64])
        >>> c
        Color(name='red', hex='#ff000040')
        >>> c.int_rgba
        (255, 0, 0, 64)

        """
        return self._red, self._green, self._blue, self._opacity

    @property
    def int_rgb(self) -> Tuple[int, int, int]:
        """Get the color value as an RGB integer tuple.

        Examples
        --------
        Create a blue color with half opacity.

        >>> import pyvista
        >>> c = pyvista.Color("blue", opacity=128)
        >>> c
        Color(name='blue', hex='#0000ff80')
        >>> c.int_rgb
        (0, 0, 255)

        Create a red color using an integer RGB sequence.

        >>> c = pyvista.Color([255, 0, 0])
        >>> c
        Color(name='red', hex='#ff0000ff')
        >>> c.int_rgb
        (255, 0, 0)

        """
        return self.int_rgba[:3]

    @property
    def float_rgba(self) -> Tuple[float, float, float, float]:
        """Get the color value as an RGBA float tuple.

        Examples
        --------
        Create a blue color with custom opacity.

        >>> import pyvista
        >>> c = pyvista.Color("blue", opacity=0.6)
        >>> c
        Color(name='blue', hex='#0000ff99')
        >>> c.float_rgba
        (0.0, 0.0, 1.0, 0.6)

        Create a transparent red color using a float RGBA sequence.

        >>> c = pyvista.Color([1.0, 0.0, 0.0, 0.2])
        >>> c
        Color(name='red', hex='#ff000033')
        >>> c.float_rgba
        (1.0, 0.0, 0.0, 0.2)

        """
        return self._red / 255.0, self._green / 255.0, self._blue / 255.0, self._opacity / 255.0

    @property
    def float_rgb(self) -> Tuple[float, float, float]:
        """Get the color value as an RGB float tuple.

        Examples
        --------
        Create a blue color with custom opacity.

        >>> import pyvista
        >>> c = pyvista.Color("blue", default_opacity=0.6)
        >>> c
        Color(name='blue', hex='#0000ff99')
        >>> c.float_rgb
        (0.0, 0.0, 1.0)

        Create a red color using a float RGB sequence.

        >>> c = pyvista.Color([1.0, 0.0, 0.0])
        >>> c
        Color(name='red', hex='#ff0000ff')
        >>> c.float_rgb
        (1.0, 0.0, 0.0)

        """
        return self.float_rgba[:3]

    @property
    def hex_rgba(self) -> str:
        """Get the color value as an RGBA hexadecimal value.

        Examples
        --------
        Create a blue color with half opacity.

        >>> import pyvista
        >>> c = pyvista.Color("blue", default_opacity="#80")
        >>> c
        Color(name='blue', hex='#0000ff80')
        >>> c.hex_rgba
        '#0000ff80'

        Create a transparent red color using an RGBA hexadecimal value.

        >>> c = pyvista.Color("0xff000040")
        >>> c
        Color(name='red', hex='#ff000040')
        >>> c.hex_rgba
        '#ff000040'

        """
        return '#' + ''.join(
            f"{c:0>2x}" for c in (self._red, self._green, self._blue, self._opacity)
        )

    @property
    def hex_rgb(self) -> str:
        """Get the color value as an RGB hexadecimal value.

        Examples
        --------
        Create a blue color with half opacity.

        >>> import pyvista
        >>> c = pyvista.Color("blue", default_opacity="#80")
        >>> c
        Color(name='blue', hex='#0000ff80')
        >>> c.hex_rgb
        '#0000ff'

        Create a red color using an RGB hexadecimal value.

        >>> c = pyvista.Color("0xff0000")
        >>> c
        Color(name='red', hex='#ff0000ff')
        >>> c.hex_rgb
        '#ff0000'

        """
        return self.hex_rgba[:-2]

    @property
    def name(self) -> Optional[str]:
        """Get the color name.

        Returns
        -------
        str or None
            The color name, in case this color has a name; otherwise ``None``.

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

        """
        return self._name

    @property
    def vtk_c3ub(self) -> _vtk.vtkColor3ub:
        """Get the color value as a VTK Color3ub instance.

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
        kwargs = f"hex={self.hex_rgba!r}"
        if self._name is not None:
            kwargs = f"name={self._name!r}, " + kwargs
        return f"Color({kwargs})"


PARAVIEW_BACKGROUND = Color('paraview').float_rgb  # [82, 87, 110] / 255


def hex_to_rgb(h):  # pragma: no cover
    """Return 0 to 1 rgb from a hex list or tuple."""
    # Deprecated on v0.34.0, estimated removal on v0.37.0
    warnings.warn(
        "The usage of `hex_to_rgb` is deprecated in favor of the new `Color` class.",
        PyVistaDeprecationWarning,
    )
    return Color(h).float_rgb


def string_to_rgb(string):  # pragma: no cover
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
    # Deprecated on v0.34.0, estimated removal on v0.37.0
    warnings.warn(
        "The usage of `string_to_rgb` is deprecated in favor of the new `Color` class.",
        PyVistaDeprecationWarning,
    )
    return Color(string).float_rgb


def get_cmap_safe(cmap):
    """Fetch a colormap by name from matplotlib, colorcet, or cmocean."""
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

        # Else use Matplotlib
        if not has_module('matplotlib'):
            raise ImportError(
                'The use of custom colormaps requires the installation of matplotlib.'
            )  # pragma: no cover

        from matplotlib import colormaps, colors

        if not isinstance(cmap, colors.Colormap):
            cmap = colormaps[cmap]

    elif isinstance(cmap, list):
        for item in cmap:
            if not isinstance(item, str):
                raise TypeError('When inputting a list as a cmap, each item should be a string.')

        if not has_module('matplotlib'):
            raise ImportError(
                'The use of custom colormaps requires the installation of matplotlib.'
            )  # pragma: no cover

        from matplotlib.colors import ListedColormap

        cmap = ListedColormap(cmap)

    return cmap
