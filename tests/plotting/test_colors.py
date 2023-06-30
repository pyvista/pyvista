import itertools

import matplotlib
import numpy as np
import pytest

import pyvista
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
    assert isinstance(get_cmap_safe(cmap), matplotlib.colors.LinearSegmentedColormap)


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
    assert pyvista.Color(name) == i_rgba
    assert pyvista.Color(name2) == i_rgba
    # Check integer types
    for i_type in i_types:
        i_color = [i_type(c) for c in i_rgba]
        # Check list, tuple and numpy array
        assert pyvista.Color(i_color) == i_rgba
        assert pyvista.Color(tuple(i_color)) == i_rgba
        assert pyvista.Color(np.asarray(i_color, dtype=i_type)) == i_rgba
    # Check float types
    for f_type in f_types:
        f_color = [f_type(c) for c in f_rgba]
        # Check list, tuple and numpy array
        assert pyvista.Color(f_color) == i_rgba
        assert pyvista.Color(tuple(f_color)) == i_rgba
        assert pyvista.Color(np.asarray(f_color, dtype=f_type)) == i_rgba
    # Check hex
    for h_prefix in h_prefixes:
        assert pyvista.Color(h_prefix + h) == i_rgba
    # Check dict
    for channels in itertools.product(*pyvista.Color.CHANNEL_NAMES):
        dct = dict(zip(channels, i_rgba))
        assert pyvista.Color(dct) == i_rgba
    # Check opacity
    for opacity in (i_opacity, f_opacity, h_opacity):
        # No opacity in color provided => use opacity
        assert pyvista.Color(name, opacity) == (*i_rgba[:3], i_opacity)
        # Opacity in color provided => overwrite using opacity
        assert pyvista.Color(i_rgba, opacity) == (*i_rgba[:3], i_opacity)
    # Check default_opacity
    for opacity in (i_opacity, f_opacity, h_opacity):
        # No opacity in color provided => use default_opacity
        assert pyvista.Color(name, default_opacity=opacity) == (*i_rgba[:3], i_opacity)
        # Opacity in color provided => keep that opacity
        assert pyvista.Color(i_rgba, default_opacity=opacity) == i_rgba
    # Check default_color
    assert pyvista.Color(None, default_color=name) == i_rgba
    # Check invalid colors and opacities
    for invalid_color in invalid_colors:
        with pytest.raises(ValueError):
            pyvista.Color(invalid_color)
    for invalid_opacity in invalid_opacities:
        with pytest.raises(ValueError):
            pyvista.Color('b', invalid_opacity)
    # Check hex and name getters
    assert pyvista.Color(name).hex_rgba == f'#{h}'
    assert pyvista.Color(name).hex_rgb == f'#{h[:-2]}'
    assert pyvista.Color('b').name == 'blue'
    # Check sRGB conversion
    assert pyvista.Color('gray', 0.5).linear_to_srgb() == '#bcbcbcbc'
    assert pyvista.Color('#bcbcbcbc').srgb_to_linear() == '#80808080'
    # Check iteration and indexing
    c = pyvista.Color(i_rgba)
    assert all(ci == fi for ci, fi in zip(c, f_rgba))
    for i, cnames in enumerate(pyvista.Color.CHANNEL_NAMES):
        assert c[i] == f_rgba[i]
        assert all(c[i] == c[cname] for cname in cnames)
    assert c[-1] == f_rgba[-1]
    assert c[1:3] == f_rgba[1:3]
    with pytest.raises(TypeError):
        c[None]  # Invalid index type
    with pytest.raises(ValueError):
        c["invalid_name"]  # Invalid string index
    with pytest.raises(IndexError):
        c[4]  # Invalid integer index


def test_color_opacity():
    color = pyvista.Color(opacity=0.5)
    assert color.opacity == 128
