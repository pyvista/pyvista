"""Native point shapes preserve topology and survive property round trips."""

from __future__ import annotations

import numpy as np
import pytest

import pyvista as pv
from pyvista.plotting._property import _HAS_NATIVE_POINT_SHAPES

pytestmark = pytest.mark.skipif(
    not _HAS_NATIVE_POINT_SHAPES, reason='Requires native point-shape rendering'
)


@pytest.mark.parametrize('shape', list(pv.PointSpriteShape))
def test_native_shape_is_property_state(shape):
    """The real property carries the shape through copy and actor replacement."""
    actor = pv.Actor()
    actor.set_point_sprite_shape(shape)
    assert actor.prop.point_shape == shape
    assert not actor._shader_replacements
    copied = actor.prop.copy()
    restored = pv.Actor(prop=copied)
    assert restored.point_sprite_shape == actor.point_sprite_shape
    actor.clear_point_sprite_shape()
    assert actor.point_sprite_shape == 'square'
    assert restored.point_sprite_shape == shape


@pytest.mark.parametrize('style', ['surface', 'wireframe', 'points'])
def test_theme_circle_reaches_rendered_vertex_cells(style):
    """An omitted shape renders circles from the plotter's own theme."""
    theme = pv.themes.Theme()
    theme.point_shape = 'circle'
    theme.multi_samples = 0
    cloud = pv.PolyData(np.array([[0.0, 0.0, 0.0]]))
    pl = pv.Plotter(theme=theme, window_size=(200, 200))
    actor = pl.add_mesh(cloud, style=style, point_size=40, lighting=False, color='white')
    pl.background_color = 'black'
    pl.camera_position = [(0, 0, 10), (0, 0, 0), (0, 1, 0)]
    pl.enable_parallel_projection()
    pl.camera.parallel_scale = 2
    pl.show(auto_close=False)
    circle = pl.screenshot()
    actor.clear_point_sprite_shape()
    pl.render()
    square = pl.screenshot()
    pl.close()
    assert circle[100, 100, 0] == square[100, 100, 0] == 255
    assert circle[82, 82, 0] == 0
    assert square[82, 82, 0] == 255
    assert 0.72 < np.count_nonzero(circle) / np.count_nonzero(square) < 0.84


def test_explicit_spheres_override_theme_circle():
    """The theme's flat shape does not disable an explicit sphere request."""
    theme = pv.themes.Theme()
    theme.point_shape = 'circle'
    pl = pv.Plotter(theme=theme)
    actor = pl.add_mesh(pv.PolyData(np.array([[0.0, 0.0, 0.0]])), render_points_as_spheres=True)
    assert actor.prop.render_points_as_spheres
    assert actor.point_sprite_shape == 'circle'
    pl.close()


def test_property_uses_its_own_theme():
    """A property created without add_mesh still resolves the theme default."""
    theme = pv.themes.Theme()
    theme.point_shape = 'diamond'
    assert pv.Property(theme=theme).point_shape == theme.point_shape


def test_invalid_native_shape_preserves_the_previous_shape():
    """Invalid input cannot silently clear the selected shape."""
    prop = pv.Property()
    prop.point_shape = 'circle'
    with pytest.raises(ValueError, match='Invalid point sprite shape'):
        prop.point_shape = 'pentagon'
    assert prop.point_shape == 'circle'
