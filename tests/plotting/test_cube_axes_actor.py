"""Test the CubeAxesActor wrapping."""

from __future__ import annotations

import numpy as np
import pytest

import pyvista as pv


@pytest.fixture
def cube_axes_actor():
    pl = pv.Plotter()
    pl.add_mesh(pv.Sphere())
    return pl.show_bounds()


def test_cube_axes_actor():
    pl = pv.Plotter()
    actor = pv.CubeAxesActor(
        pl.camera,
        x_label_format=None,
        y_label_format=None,
        z_label_format=None,
    )
    assert isinstance(actor.camera, pv.Camera)

    # ensure label format is set to default
    expected_fmt = '%.1f' if pv.vtk_version_info < (9, 6, 0) else '{0:.1f}'
    assert actor.x_label_format == expected_fmt
    assert actor.y_label_format == expected_fmt
    assert actor.z_label_format == expected_fmt


def test_labels(cube_axes_actor):
    # test setting "format" to just a string
    cube_axes_actor.x_label_format = 'Value'
    assert len(cube_axes_actor.x_labels) == 5
    assert set(cube_axes_actor.x_labels) == {'Value'}

    # test no format, values should match exactly
    cube_axes_actor.y_label_format = ''
    assert len(cube_axes_actor.y_labels) == 5
    values = np.array(cube_axes_actor.y_labels, float)
    expected = np.linspace(cube_axes_actor.bounds.y_min, cube_axes_actor.bounds.y_max, 5)
    assert np.allclose(values, expected)

    # standard format
    cube_axes_actor.z_label_format = '%.1f' if pv.vtk_version_info < (9, 6, 0) else '{0:.1f}'
    assert len(cube_axes_actor.z_labels) == 5
    assert all(len(label) < 5 for label in cube_axes_actor.z_labels)


def test_tick_location(cube_axes_actor):
    assert isinstance(cube_axes_actor.tick_location, str)

    for location in ['inside', 'outside', 'both']:
        cube_axes_actor.tick_location = location
        assert cube_axes_actor.tick_location == location


def test_use_2d_mode(cube_axes_actor):
    assert isinstance(cube_axes_actor.use_2d_mode, bool)
    cube_axes_actor.use_2d_mode = False
    assert cube_axes_actor.use_2d_mode is False


def test_label_visibility_setter(cube_axes_actor):
    assert isinstance(cube_axes_actor.x_label_visibility, bool)
    cube_axes_actor.x_label_visibility = False
    assert cube_axes_actor.x_label_visibility is False

    assert isinstance(cube_axes_actor.y_label_visibility, bool)
    cube_axes_actor.y_label_visibility = False
    assert cube_axes_actor.y_label_visibility is False

    assert isinstance(cube_axes_actor.z_label_visibility, bool)
    cube_axes_actor.z_label_visibility = False
    assert cube_axes_actor.z_label_visibility is False


def test_titles(cube_axes_actor):
    assert isinstance(cube_axes_actor.x_title, str)
    cube_axes_actor.x_title = 'x foo'
    assert cube_axes_actor.x_title == 'x foo'

    assert isinstance(cube_axes_actor.y_title, str)
    cube_axes_actor.y_title = 'y foo'
    assert cube_axes_actor.y_title == 'y foo'

    assert isinstance(cube_axes_actor.z_title, str)
    cube_axes_actor.z_title = 'z foo'
    assert cube_axes_actor.z_title == 'z foo'


@pytest.mark.parametrize('title', ['x_title', 'y_title', 'z_title'])
def test_title_must_be_string(cube_axes_actor, title):
    with pytest.raises(TypeError, match=rf'{title} must be an instance of .*str'):
        setattr(cube_axes_actor, title, None)


def test_axis_minor_tick_visibility(cube_axes_actor):
    assert isinstance(cube_axes_actor.x_axis_minor_tick_visibility, bool)
    cube_axes_actor.x_axis_minor_tick_visibility = False
    assert cube_axes_actor.x_axis_minor_tick_visibility is False

    assert isinstance(cube_axes_actor.y_axis_minor_tick_visibility, bool)
    cube_axes_actor.y_axis_minor_tick_visibility = False
    assert cube_axes_actor.y_axis_minor_tick_visibility is False

    assert isinstance(cube_axes_actor.z_axis_minor_tick_visibility, bool)
    cube_axes_actor.z_axis_minor_tick_visibility = False
    assert cube_axes_actor.z_axis_minor_tick_visibility is False


def test_title_offset_sequence(cube_axes_actor):
    assert isinstance(cube_axes_actor.title_offset, tuple)
    cube_axes_actor.title_offset = (t := (0.01, 0.02))
    assert cube_axes_actor.title_offset == t


def test_label_offset(cube_axes_actor):
    assert isinstance(cube_axes_actor.label_offset, float)
    cube_axes_actor.label_offset = 0.01
    assert cube_axes_actor.label_offset == 0.01


@pytest.fixture
def camera():
    return pv.Plotter().camera


def test_color_default(camera):
    actor = pv.CubeAxesActor(camera)
    expected = pv.Color(pv.global_theme.font.color).float_rgb
    assert actor.GetXAxesLinesProperty().GetColor() == expected
    assert actor.GetYAxesLinesProperty().GetColor() == expected
    assert actor.GetZAxesLinesProperty().GetColor() == expected
    assert actor.GetTitleTextProperty(0).GetColor() == expected
    assert actor.GetLabelTextProperty(0).GetColor() == expected


def test_color(camera):
    actor = pv.CubeAxesActor(camera, color='red', grid=True)
    expected = pv.Color('red').float_rgb
    assert actor.GetXAxesLinesProperty().GetColor() == expected
    assert actor.GetXAxesGridlinesProperty().GetColor() == expected
    assert actor.GetTitleTextProperty(2).GetColor() == expected


@pytest.mark.parametrize(
    ('grid', 'expected'),
    [
        (True, pv.CubeAxesActor.VTK_GRID_LINES_FURTHEST),
        ('back', pv.CubeAxesActor.VTK_GRID_LINES_FURTHEST),
        ('backface', pv.CubeAxesActor.VTK_GRID_LINES_FURTHEST),
        ('front', pv.CubeAxesActor.VTK_GRID_LINES_CLOSEST),
        ('frontface', pv.CubeAxesActor.VTK_GRID_LINES_CLOSEST),
        ('all', pv.CubeAxesActor.VTK_GRID_LINES_ALL),
        ('both', pv.CubeAxesActor.VTK_GRID_LINES_ALL),
    ],
)
def test_grid(camera, grid, expected):
    actor = pv.CubeAxesActor(camera, grid=grid)
    assert actor.GetGridLineLocation() == expected
    assert actor.GetDrawXGridlines()
    assert actor.GetDrawYGridlines()
    assert actor.GetDrawZGridlines()


def test_grid_follows_axis_visibility(camera):
    actor = pv.CubeAxesActor(camera, grid=True, y_axis_visibility=False)
    assert actor.GetDrawXGridlines()
    assert not actor.GetDrawYGridlines()


def test_grid_none(camera):
    actor = pv.CubeAxesActor(camera)
    assert not actor.GetDrawXGridlines()


def test_grid_raises(camera):
    with pytest.raises(TypeError, match='`grid` must be a str'):
        pv.CubeAxesActor(camera, grid=1.0)
    with pytest.raises(ValueError, match='`grid` must be either'):
        pv.CubeAxesActor(camera, grid='sideways')


@pytest.mark.parametrize(
    ('location', 'expected'),
    [
        ('all', pv.CubeAxesActor.VTK_FLY_STATIC_EDGES),
        ('origin', pv.CubeAxesActor.VTK_FLY_STATIC_TRIAD),
        ('outer', pv.CubeAxesActor.VTK_FLY_OUTER_EDGES),
        ('default', pv.CubeAxesActor.VTK_FLY_CLOSEST_TRIAD),
        ('closest', pv.CubeAxesActor.VTK_FLY_CLOSEST_TRIAD),
        ('front', pv.CubeAxesActor.VTK_FLY_CLOSEST_TRIAD),
        ('furthest', pv.CubeAxesActor.VTK_FLY_FURTHEST_TRIAD),
        ('back', pv.CubeAxesActor.VTK_FLY_FURTHEST_TRIAD),
    ],
)
def test_location(camera, location, expected):
    assert pv.CubeAxesActor(camera, location=location).GetFlyMode() == expected


def test_location_default(camera):
    assert pv.CubeAxesActor(camera).GetFlyMode() == pv.CubeAxesActor.VTK_FLY_CLOSEST_TRIAD


def test_location_raises(camera):
    with pytest.raises(TypeError, match='location must be a string'):
        pv.CubeAxesActor(camera, location=1)
    with pytest.raises(ValueError, match='Value of location'):
        pv.CubeAxesActor(camera, location='sideways')


def test_font_2d_text(camera):
    actor = pv.CubeAxesActor(
        camera, font_size=42, font_family='times', bold=False, use_3d_text=False
    )
    prop = actor.GetTitleTextProperty(0)
    assert prop.GetFontSize() == 42
    assert prop.GetFontFamilyAsString() == 'Times'
    assert not prop.GetBold()


def test_font_3d_text(camera):
    """3D text renders at a fixed high resolution and is scaled down by the screen size."""
    actor = pv.CubeAxesActor(camera, font_size=42, use_3d_text=True)
    assert actor.GetTitleTextProperty(0).GetFontSize() == 50
    factor = 1.0 if pv.vtk_version_info < (9, 6, 0) else 50 / 12
    assert actor.GetScreenSize() == pytest.approx(42 / 12 / factor * 10)


def test_font_defaults(camera):
    actor = pv.CubeAxesActor(camera, use_3d_text=False)
    prop = actor.GetLabelTextProperty(1)
    assert prop.GetFontSize() == pv.global_theme.font.size
    assert prop.GetBold()


def test_use_3d_text_default(camera):
    expected = pv.vtk_version_info < (9, 6, 0)
    assert bool(pv.CubeAxesActor(camera).GetUseTextActor3D()) is expected


@pytest.mark.parametrize('use_3d_text', [True, False])
def test_use_3d_text(camera, use_3d_text):
    actor = pv.CubeAxesActor(camera, use_3d_text=use_3d_text)
    assert bool(actor.GetUseTextActor3D()) is use_3d_text


def test_use_2d_mode_init(camera):
    assert pv.CubeAxesActor(camera, use_2d_mode=True).use_2d_mode is True
    assert pv.CubeAxesActor(camera).use_2d_mode is False


def test_bounds_init(camera):
    bounds = (-1, 2, -3, 4, -5, 6)
    actor = pv.CubeAxesActor(camera, bounds=bounds)
    assert actor.bounds == bounds
    assert actor.x_labels[0] == '-1.0'
    assert actor.x_labels[-1] == '2.0'


def test_padding(camera):
    actor = pv.CubeAxesActor(camera, bounds=(0, 10, 0, 10, 0, 10), padding=0.1)
    assert actor.bounds == (-1, 11, -1, 11, -1, 11)


def test_padding_raises(camera):
    with pytest.raises(ValueError, match='padding'):
        pv.CubeAxesActor(camera, bounds=(0, 1, 0, 1, 0, 1), padding=1.5)


def test_axes_ranges_init(camera):
    actor = pv.CubeAxesActor(camera, bounds=(0, 1, 0, 1, 0, 1), axes_ranges=(0, 10, 0, 20, 0, 30))
    assert actor.x_axis_range == (0, 10)
    assert actor.y_axis_range == (0, 20)
    assert actor.z_axis_range == (0, 30)
    assert actor.z_labels[-1] == '30.0'


def test_axes_ranges_init_raises(camera):
    with pytest.raises(ValueError, match='axes_ranges must have a length equal to any of: 6'):
        pv.CubeAxesActor(camera, axes_ranges=1)
    with pytest.raises(TypeError, match='axes_ranges must have real numbers'):
        pv.CubeAxesActor(camera, axes_ranges=[0, 1, 'a', 'b', 2, 3])
    with pytest.raises(ValueError, match='Got length 5 instead'):
        pv.CubeAxesActor(camera, axes_ranges=[0, 1, 2, 3, 4])
