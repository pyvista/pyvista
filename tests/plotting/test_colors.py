from __future__ import annotations

import colorsys
import importlib.util
import itertools
import re

import cmcrameri
import cmocean
import colorcet
import matplotlib as mpl
from matplotlib.colors import CSS4_COLORS
from matplotlib.colors import TABLEAU_COLORS
import numpy as np
import pytest

import pyvista as pv
from pyvista.plotting import _vtk
from pyvista.plotting.colors import _CMCRAMERI_CMAPS
from pyvista.plotting.colors import _CMOCEAN_CMAPS
from pyvista.plotting.colors import _COLORCET_CMAPS
from pyvista.plotting.colors import _MATPLOTLIB_CMAPS
from pyvista.plotting.colors import _format_color_name
from pyvista.plotting.colors import color_scheme_to_cycler
from pyvista.plotting.colors import get_cmap_safe

COLORMAPS = ['Greys', mpl.colormaps['viridis'], ['red', 'green', 'blue']]

if importlib.util.find_spec('cmocean'):
    COLORMAPS.append('algae')


if importlib.util.find_spec('colorcet'):
    COLORMAPS.append('fire')

if importlib.util.find_spec('cmcrameri'):
    COLORMAPS.append('batlow')


@pytest.mark.parametrize('cmap', COLORMAPS)
def test_get_cmap_safe(cmap):
    assert isinstance(get_cmap_safe(cmap), mpl.colors.Colormap)


@pytest.mark.parametrize('scheme', [object(), 1.0, None])
def test_color_scheme_to_cycler_raises(scheme):
    with pytest.raises(TypeError, match=f'Color scheme not understood: {scheme}'):
        color_scheme_to_cycler(scheme=scheme)


def test_color():
    name, name2 = 'blue', 'b'
    i_rgba, f_rgba = (0, 0, 255, 255), (0.0, 0.0, 1.0, 1.0)
    h = '0000ffff'
    i_opacity, f_opacity, h_opacity = 153, 0.6, '99'
    i_types = (int, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)
    f_types = (float, np.float16, np.float32, np.float64)
    h_prefixes = ('', '0x', '#')
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
        dct = dict(zip(channels, i_rgba, strict=True))
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
    # Check hex and name getters
    assert pv.Color(name).hex_rgba == f'#{h}'
    assert pv.Color(name).hex_rgb == f'#{h[:-2]}'
    assert pv.Color('b').name == 'blue'
    # Check sRGB conversion
    assert pv.Color('gray', 0.5).linear_to_srgb() == '#bcbcbcbc'
    assert pv.Color('#bcbcbcbc').srgb_to_linear() == '#80808080'
    # Check iteration and indexing
    c = pv.Color(i_rgba)
    assert all(ci == fi for ci, fi in zip(c, f_rgba, strict=True))
    for i, cnames in enumerate(pv.Color.CHANNEL_NAMES):
        assert c[i] == f_rgba[i]
        assert all(c[i] == c[cname] for cname in cnames)
    assert c[-1] == f_rgba[-1]
    assert c[1:3] == f_rgba[1:3]
    with pytest.raises(TypeError):
        c[None]  # Invalid index type
    with pytest.raises(ValueError):  # noqa: PT011
        c['invalid_name']  # Invalid string index
    with pytest.raises(IndexError):
        c[4]  # Invalid integer index


@pytest.mark.parametrize('opacity', [275, -50, 2.4, -1.2, '#zz'])
def test_color_invalid_opacity(opacity):
    match = (
        'Must be an integer, float or string.  For example:'
        "\n\t\topacity='1.0'"
        "\n\t\topacity='255'"
        "\n\t\topacity='#FF'"
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        pv.Color('b', opacity)


@pytest.mark.parametrize(
    'color',
    [
        (300, 0, 0),
        (0, -10, 0),
        (0, 0, 1.5),
        np.array((0, 0, 1.5), dtype=np.float16),
        (-0.5, 0, 0),
        (0, 0),
        '#hh0000',
        'invalid_name',
        {'invalid_name': 100},
    ],
)
def test_color_invalid_values(color):
    match = (
        'Must be a string, rgb(a) sequence, or hex color string.  For example:'
        "\n\t\tcolor='white'"
        "\n\t\tcolor='w'"
        '\n\t\tcolor=[1.0, 1.0, 1.0]'
        '\n\t\tcolor=[255, 255, 255]'
        "\n\t\tcolor='#FFFFFF'"
    )
    with pytest.raises(ValueError, match=re.escape(match)):
        pv.Color(color)


def test_color_invalid_type():
    match = 'color must be an instance of'
    with pytest.raises(TypeError, match=match):
        pv.Color(range(3))


@pytest.mark.parametrize('delimiter', ['-', '_', ' '])
def test_color_name_delimiter(delimiter):
    name = f'medium{delimiter}spring{delimiter}green'
    c = pv.Color(name)
    assert c.name == name.replace(delimiter, '_')


def test_color_hls():
    lime = pv.Color('lime')
    actual_hls = lime._float_hls
    expected_hls = colorsys.rgb_to_hls(*lime.float_rgb)
    assert actual_hls == expected_hls


def test_color_opacity():
    color = pv.Color(opacity=0.5)
    assert color.opacity == 128


def pytest_generate_tests(metafunc):
    """Generate parametrized tests."""
    if 'css4_color' in metafunc.fixturenames:
        color_names = list(CSS4_COLORS.keys())
        color_values = list(CSS4_COLORS.values())

        test_cases = zip(color_names, color_values, strict=True)
        metafunc.parametrize('css4_color', test_cases, ids=color_names)

    if 'tab_color' in metafunc.fixturenames:
        color_names = list(TABLEAU_COLORS.keys())
        color_values = list(TABLEAU_COLORS.values())

        test_cases = zip(color_names, color_values, strict=True)
        metafunc.parametrize('tab_color', test_cases, ids=color_names)

    if 'vtk_color' in metafunc.fixturenames:
        color_names = list(pv.plotting.colors._VTK_COLORS.keys())
        color_values = list(pv.plotting.colors._VTK_COLORS.values())

        test_cases = zip(color_names, color_values, strict=True)
        metafunc.parametrize('vtk_color', test_cases, ids=color_names)

    if 'paraview_color' in metafunc.fixturenames:
        color_names = list(pv.plotting.colors._PARAVIEW_COLORS.keys())
        color_values = list(pv.plotting.colors._PARAVIEW_COLORS.values())

        test_cases = zip(color_names, color_values, strict=True)
        metafunc.parametrize('paraview_color', test_cases, ids=color_names)

    if 'color_synonym' in metafunc.fixturenames:
        synonyms = list(pv.colors._formatted_color_synonyms.keys())
        metafunc.parametrize('color_synonym', synonyms, ids=synonyms)


@pytest.mark.skip_check_gc
def test_css4_colors(css4_color):
    # Test value
    name, value = css4_color
    color = pv.Color(name)
    assert color.hex_rgb.lower() == value.lower()

    # Test name
    assert color.name in pv.plotting.colors._CSS_COLORS

    if _format_color_name(color.name) != name:
        # Must be a synonym
        assert name in pv.plotting.colors._formatted_color_synonyms


@pytest.mark.skip_check_gc
def test_tab_colors(tab_color):
    # Test value
    name, value = tab_color
    assert pv.Color(name).hex_rgb.lower() == value.lower()

    # Test name
    assert name in pv.plotting.colors._TABLEAU_COLORS


@pytest.mark.skip_check_gc
def test_vtk_colors(vtk_color):
    name, value = vtk_color

    # Some pyvista colors are technically not valid VTK colors. We need to map their
    # synonym manually for the tests
    vtk_synonyms = {  # pyvista_color : vtk_color
        'light_slate_blue': 'slate_blue_light',
        'deep_cadmium_red': 'cadmium_red_deep',
        'light_cadmium_red': 'cadmium_red_light',
        'light_cadmium_yellow': 'cadmium_yellow_light',
        'deep_cobalt_violet': 'cobalt_violet_deep',
        'deep_naples_yellow': 'naples_yellow_deep',
        'light_viridian': 'viridian_light',
    }
    name = vtk_synonyms.get(name, name)
    expected_hex = _vtk_named_color_as_hex(name)
    assert value.lower() == expected_hex


def _vtk_named_color_as_hex(name: str) -> str:
    # Get expected hex value from vtkNamedColors
    color3ub = _vtk.vtkNamedColors().GetColor3ub(name)
    int_rgb = (color3ub.GetRed(), color3ub.GetGreen(), color3ub.GetBlue())
    if int_rgb == (0.0, 0.0, 0.0) and name != 'black':
        pytest.fail(f"Color '{name}' is not a valid VTK color.")
    return pv.Color(int_rgb).hex_rgb


@pytest.mark.skip_check_gc
@pytest.mark.needs_vtk_version(9, 6, 99)  # >= 9.7.0
def test_paraview_colors(paraview_color):
    name, value = paraview_color

    # Map PyVista color names to names used by vtkNamedColors
    paraview_map = {
        'paraview_background': 'ParaViewBlueGrayBkg',
        'paraview_background_warm': 'ParaViewWarmGrayBkg',
    }
    name = paraview_map[name]
    expected_hex = _vtk_named_color_as_hex(name)
    assert value.lower() == expected_hex


@pytest.mark.skip_check_gc
def test_color_synonyms(color_synonym):
    color = pv.Color(color_synonym)
    assert isinstance(color, pv.Color)


def test_unique_colors():
    duplicates = np.rec.find_duplicate(pv.hex_colors.values())
    if len(duplicates) > 0:
        pytest.fail(f'The following colors have duplicate definitions: {duplicates}.')


@pytest.fixture
def reset_matplotlib_cmaps():
    # Need to unregister all 3rd-party cmaps
    for cmap in list(mpl.colormaps):
        try:
            mpl.colormaps.unregister(cmap)
        except (ValueError, AttributeError):
            continue


def maybe_xfail_mpl():
    missing_colormaps = {'berlin', 'vanimo', 'managua', 'okabe_ito'}
    if not missing_colormaps.issubset(mpl.colormaps):
        pytest.xfail(
            reason=f'Older Matplotlib is missing colormaps: {missing_colormaps}.',
        )


@pytest.mark.usefixtures('reset_matplotlib_cmaps')
def test_cmaps_matplotlib_allowed():
    maybe_xfail_mpl()
    # Test that cmaps listed in colors module matches the actual cmaps available
    actual = set(mpl.colormaps)
    expected = set(_MATPLOTLIB_CMAPS)
    assert actual == expected


@pytest.mark.usefixtures('reset_matplotlib_cmaps')
def test_cmaps_colorcet_required():
    # Test that cmaps listed in colors module matches the actual cmaps available
    actual = set(colorcet.cm.keys()) - set(mpl.colormaps)
    expected = set(_COLORCET_CMAPS)
    assert actual == expected


@pytest.mark.usefixtures('reset_matplotlib_cmaps')
def test_cmaps_cmocean_required():
    # Test that cmaps listed in colors module matches the actual cmaps available
    actual = set(cmocean.cm.cmap_d.keys()) - set(mpl.colormaps)
    expected = set(_CMOCEAN_CMAPS)
    assert actual == expected


@pytest.mark.usefixtures('reset_matplotlib_cmaps')
def test_cmaps_cmcrameri_required():
    maybe_xfail_mpl()
    # Test that cmaps listed in colors module matches the actual cmaps available
    actual = set(cmcrameri.cm.cmaps.keys()) - set(mpl.colormaps)
    expected = set(_CMCRAMERI_CMAPS)
    assert actual == expected
