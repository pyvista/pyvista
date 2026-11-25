from __future__ import annotations

import numpy as np
import pytest

import pyvista as pv


@pytest.fixture
def lookup_theme():
    theme = pv.themes.Theme()
    theme.cmap = 'plasma'
    theme.nan_color = pv.Color('cyan', opacity=0.5)
    theme.above_range_color = 'orange'
    theme.below_range_color = 'purple'
    return theme


def test_lookup_table_global_theme(lookup_theme):
    pv.global_theme.load_theme(lookup_theme)
    lut = pv.LookupTable()
    assert lut.cmap.name == 'plasma'
    assert lut.nan_color == lookup_theme.nan_color
    assert lut.nan_opacity == 128
    assert lut.above_range_color is None
    assert lut.below_range_color is None


def test_lookup_table_theme(lookup_theme):
    pv.global_theme.cmap = 'gray'
    pv.global_theme.nan_color = 'red'
    lut = pv.LookupTable(theme=lookup_theme, above_range_color=True, below_range_color=True)
    assert lut.cmap.name == 'plasma'
    assert lut.nan_color == lookup_theme.nan_color
    assert lut.above_range_color == lookup_theme.above_range_color
    assert lut.below_range_color == lookup_theme.below_range_color
    assert pv.global_theme.cmap == 'gray'
    assert pv.global_theme.nan_color == pv.Color('red')


def test_lookup_table_theme_overrides(lookup_theme):
    lut = pv.LookupTable(
        theme=lookup_theme,
        cmap='gray',
        n_values=2,
        flip=True,
        nan_color='red',
        above_range_color='blue',
        below_range_color=False,
    )
    assert lut.cmap.name == 'gray'
    np.testing.assert_array_equal(lut.values, [[255, 255, 255, 255], [0, 0, 0, 255]])
    assert lut.nan_color == pv.Color('red')
    assert lut.above_range_color == pv.Color('blue')
    assert lut.below_range_color is None
    assert lookup_theme.cmap == 'plasma'
    assert lookup_theme.nan_color == pv.Color('cyan', opacity=0.5)


def test_lookup_table_theme_values(lookup_theme):
    values = [[10, 20, 30, 40], [50, 60, 70, 80]]
    lut = pv.LookupTable(values=values, theme=lookup_theme)
    np.testing.assert_array_equal(lut.values, values)
    assert lut.cmap is None
    assert lut.n_values == 2
    assert lut.nan_color == lookup_theme.nan_color


@pytest.mark.parametrize('opacity', [1.0, 255])
def test_lookup_table_theme_nan_opacity(lookup_theme, opacity):
    lut = pv.LookupTable(theme=lookup_theme)
    lut.nan_opacity = opacity
    assert lut.nan_color == pv.Color('cyan')
    assert lookup_theme.nan_color.opacity == 128


@pytest.mark.parametrize(
    ('name', 'value'),
    [
        ('value_range', (0.1, 0.2)),
        ('hue_range', (0.2, 0.3)),
        ('alpha_range', (0.3, 0.4)),
        ('ramp', 'linear'),
    ],
)
def test_lookup_table_theme_custom_table(lookup_theme, name, value):
    lut = pv.LookupTable(theme=lookup_theme, n_values=17, **{name: value})
    assert lut.cmap is None
    assert lut.n_values == 17
    assert getattr(lut, name) == value


@pytest.mark.parametrize('range_name', ['above', 'below'])
@pytest.mark.parametrize('setting', ['color', 'opacity'])
def test_lookup_table_theme_range_colors(lookup_theme, range_name, setting):
    name = f'{range_name}_range_color'
    color = getattr(lookup_theme, name)
    lut = pv.LookupTable(theme=lookup_theme)
    setattr(lookup_theme, name, 'yellow')
    setattr(pv.global_theme, name, 'blue')
    setattr(lut, f'{range_name}_range_{setting}', True if setting == 'color' else 0.5)
    expected = color if setting == 'color' else pv.Color(color, opacity=0.5)
    assert getattr(lut, name) == expected
    setattr(lut, name, False)
    assert getattr(lut, name) is None


def test_lookup_table_theme_snapshot(lookup_theme):
    lut = pv.LookupTable(theme=lookup_theme)
    values = lut.values.copy()
    lookup_theme.cmap = 'gray'
    lookup_theme.nan_color = 'red'
    assert lut.cmap.name == 'plasma'
    np.testing.assert_array_equal(lut.values, values)
    assert lut.nan_color == pv.Color('cyan', opacity=0.5)


def test_lookup_table_invalid_theme():
    with pytest.raises(TypeError, match='must be a pyvista theme'):
        pv.LookupTable(theme=123)


@pytest.mark.parametrize(
    'mapper_type', [pv.DataSetMapper, pv.PointGaussianMapper, pv.SmartVolumeMapper]
)
def test_mapper_lookup_table_theme(lookup_theme, mapper_type):
    pv.global_theme.cmap = 'gray'
    pv.global_theme.nan_color = 'red'
    mapper = mapper_type(theme=lookup_theme)
    assert mapper.lookup_table.cmap.name == 'plasma'
    assert mapper.lookup_table.nan_color == lookup_theme.nan_color
    copied = mapper.copy()
    assert copied.lookup_table.cmap.name == 'plasma'
    assert copied.lookup_table.nan_color == lookup_theme.nan_color
