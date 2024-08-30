from __future__ import annotations

import itertools
from typing import final

import matplotlib as mpl
import numpy as np
import pytest

import pyvista as pv
from pyvista.plotting.colors import get_cmap_safe

COLORMAPS = ['Greys']

try:
    import cmocean  # noqa: F401

    COLORMAPS.append('algae')
except ImportError:
    pass


try:
    import colorcet  # noqa: F401

    COLORMAPS.append('fire')
except:
    pass


@pytest.mark.parametrize("cmap", COLORMAPS)
def test_get_cmap_safe(cmap):
    assert isinstance(get_cmap_safe(cmap), mpl.colors.LinearSegmentedColormap)


def test_color():
    name, name2 = "blue", "b"
    i_rgba, f_rgba = (0, 0, 255, 255), (0.0, 0.0, 1.0, 1.0)
    h = "0000ffff"
    i_opacity, f_opacity, h_opacity = 153, 0.6, "99"
    invalid_colors = (
        (300, 0, 0),
        (0, -10, 0),
        (0, 0, 1.5),
        (-0.5, 0, 0),
        (0, 0),
        "#hh0000",
        "invalid_name",
        {"invalid_name": 100},
    )
    invalid_opacities = (275, -50, 2.4, -1.2, "#zz")
    i_types = (int, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)
    f_types = (float, np.float16, np.float32, np.float64)
    h_prefixes = ("", "0x", "#")
    assert pv.Color(name) == i_rgba
    assert pv.Color(name2) == i_rgba
    # Check integer types
    for i_type in i_types:
        i_color = [i_type(c) for c in i_rgba]
        # Check list, tuple and numpy array
        assert pv.Color(i_color) == i_rgba
        assert pv.Color(tuple(i_color)) == i_rgba
        assert pv.Color(np.asarray(i_color, dtype=i_type)) == i_rgba
    # Check float types
    for f_type in f_types:
        f_color = [f_type(c) for c in f_rgba]
        # Check list, tuple and numpy array
        assert pv.Color(f_color) == i_rgba
        assert pv.Color(tuple(f_color)) == i_rgba
        assert pv.Color(np.asarray(f_color, dtype=f_type)) == i_rgba
    # Check hex
    for h_prefix in h_prefixes:
        assert pv.Color(h_prefix + h) == i_rgba
    # Check dict
    for channels in itertools.product(*pv.Color.CHANNEL_NAMES):
        dct = dict(zip(channels, i_rgba))
        assert pv.Color(dct) == i_rgba
    # Check opacity
    for opacity in (i_opacity, f_opacity, h_opacity):
        # No opacity in color provided => use opacity
        assert pv.Color(name, opacity) == (*i_rgba[:3], i_opacity)
        # Opacity in color provided => overwrite using opacity
        assert pv.Color(i_rgba, opacity) == (*i_rgba[:3], i_opacity)
    # Check default_opacity
    for opacity in (i_opacity, f_opacity, h_opacity):
        # No opacity in color provided => use default_opacity
        assert pv.Color(name, default_opacity=opacity) == (*i_rgba[:3], i_opacity)
        # Opacity in color provided => keep that opacity
        assert pv.Color(i_rgba, default_opacity=opacity) == i_rgba
    # Check default_color
    assert pv.Color(None, default_color=name) == i_rgba
    # Check invalid colors and opacities
    for invalid_color in invalid_colors:
        with pytest.raises(ValueError):  # noqa: PT011
            pv.Color(invalid_color)
    for invalid_opacity in invalid_opacities:
        with pytest.raises(ValueError):  # noqa: PT011
            pv.Color('b', invalid_opacity)
    # Check hex and name getters
    assert pv.Color(name).hex_rgba == f'#{h}'
    assert pv.Color(name).hex_rgb == f'#{h[:-2]}'
    assert pv.Color('b').name == 'blue'
    # Check sRGB conversion
    assert pv.Color('gray', 0.5).linear_to_srgb() == '#bcbcbcbc'
    assert pv.Color('#bcbcbcbc').srgb_to_linear() == '#80808080'
    # Check iteration and indexing
    c = pv.Color(i_rgba)
    assert all(ci == fi for ci, fi in zip(c, f_rgba))
    for i, cnames in enumerate(pv.Color.CHANNEL_NAMES):
        assert c[i] == f_rgba[i]
        assert all(c[i] == c[cname] for cname in cnames)
    assert c[-1] == f_rgba[-1]
    assert c[1:3] == f_rgba[1:3]
    with pytest.raises(TypeError):
        c[None]  # Invalid index type
    with pytest.raises(ValueError):  # noqa: PT011
        c["invalid_name"]  # Invalid string index
    with pytest.raises(IndexError):
        c[4]  # Invalid integer index


def test_color_opacity():
    color = pv.Color(opacity=0.5)
    assert color.opacity == 128


# Define standardized names and hex values of web colors defined here:
# https://www.w3.org/TR/css-color-4/#named-colors
WEB_COLORS: final = {
    'aliceblue': '#F0F8FF',
    'antiquewhite': '#FAEBD7',
    'aqua': '#00FFFF',
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
    'darkgrey': '#A9A9A9',
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
    'darkslategrey': '#2F4F4F',
    'darkturquoise': '#00CED1',
    'darkviolet': '#9400D3',
    'deeppink': '#FF1493',
    'deepskyblue': '#00BFFF',
    'dimgray': '#696969',
    'dimgrey': '#696969',
    'dodgerblue': '#1E90FF',
    'firebrick': '#B22222',
    'floralwhite': '#FFFAF0',
    'forestgreen': '#228B22',
    'fuchsia': '#FF00FF',
    'gainsboro': '#DCDCDC',
    'ghostwhite': '#F8F8FF',
    'gold': '#FFD700',
    'goldenrod': '#DAA520',
    'gray': '#808080',
    'green': '#008000',
    'greenyellow': '#ADFF2F',
    'grey': '#808080',
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
    'lightgrey': '#D3D3D3',
    'lightpink': '#FFB6C1',
    'lightsalmon': '#FFA07A',
    'lightseagreen': '#20B2AA',
    'lightskyblue': '#87CEFA',
    'lightslategray': '#778899',
    'lightslategrey': '#778899',
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
    'slategrey': '#708090',
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
    # 'tab:blue': '#1f77b4',
    # 'tab:orange': '#ff7f0e',
    # 'tab:green': '#2ca02c',
    # 'tab:red': '#d62728',
    # 'tab:purple': '#9467bd',
    # 'tab:brown': '#8c564b',
    # 'tab:pink': '#e377c2',
    # 'tab:gray': '#7f7f7f',
    # 'tab:olive': '#bcbd22',
    # 'tab:cyan': '#17becf',
    # # Char synonyms
    # 'b': 'blue',
    # 'g': 'green',
    # 'r': 'red',
    # 'c': 'cyan',
    # 'm': 'magenta',
    # 'y': 'yellow',
    # 'k': 'black',
    # 'w': 'white',
    # Named synonyms
    #     'aqua': 'cyan',
    #     'darkgrey': 'darkgray',
    #     'darkslategrey': 'darkslategray',
    #     'dimgrey': 'dimgray',
    #     'fuchsia': 'magenta',
    #     'grey': 'gray',
    #     'lightgrey': 'lightgray',
    #     'lightslategrey': 'lightslategray',
    #     'pv': 'paraview_background',
    #     'paraview': 'paraview_background',
    #     'slategrey': 'slategray',
    #
}


def pytest_generate_tests(metafunc):
    """Generate parametrized tests."""
    if 'web_color' in metafunc.fixturenames:
        color_names = list(WEB_COLORS.keys())
        color_values = list(WEB_COLORS.values())

        test_cases = zip(color_names, color_values)
        metafunc.parametrize('web_color', test_cases, ids=color_names)

    if 'color_synonym' in metafunc.fixturenames:
        synonyms = list(pv.colors.color_synonyms.keys())
        metafunc.parametrize('color_synonym', synonyms, ids=synonyms)


def test_web_colors(web_color):
    name, value = web_color
    assert pv.Color(name).hex_rgb.lower() == value.lower()


def test_color_synonyms(color_synonym):
    color = pv.Color(color_synonym)
    assert color.name


def test_unique_colors():
    duplicates = np.rec.find_duplicate(pv.hexcolors.values())
    if len(duplicates) > 0:
        pytest.fail(f"The following colors have duplicate definitions: {duplicates}.")
