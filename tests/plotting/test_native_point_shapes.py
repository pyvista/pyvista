"""Native point shapes preserve topology and survive property round trips."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import pyvista as pv
from pyvista.plotting._property import _HAS_NATIVE_POINT_SHAPES


@pytest.fixture
def native_property_api(monkeypatch):
    """Exercise Python wrapper contracts independently of native rendering."""
    # mocked: stock VTK lacks the extended enum/API. Substitute only that binding
    # for wrapper tests; rendering and native DeepCopy tests never use this shim.
    shapes = {}
    names = ('Round', 'Square', 'Triangle', 'Hexagon', 'Diamond', 'Asterisk', 'Star')
    monkeypatch.setattr(
        pv.Property,
        'Point2DShapeType',
        SimpleNamespace(**{name: index for index, name in enumerate(names)}),
        raising=False,
    )
    monkeypatch.setattr(
        pv.Property, 'GetPoint2DShape', lambda self: shapes[self.__this__], raising=False
    )
    monkeypatch.setattr(
        pv.Property,
        'SetPoint2DShape',
        lambda self, value: shapes.__setitem__(self.__this__, value),
        raising=False,
    )
    for module in ('_property', 'actor', '_plotting'):
        monkeypatch.setattr(f'pyvista.plotting.{module}._HAS_NATIVE_POINT_SHAPES', True)


@pytest.mark.parametrize(
    ('shape', 'native_name'),
    [
        ('circle', 'Round'),
        ('triangle', 'Triangle'),
        ('hexagon', 'Hexagon'),
        ('diamond', 'Diamond'),
        ('asterisk', 'Asterisk'),
        ('star', 'Star'),
    ],
)
@pytest.mark.usefixtures('native_property_api')
@pytest.mark.parametrize('as_enum', [False, True])
def test_native_shape_is_property_state(shape, native_name, as_enum):
    """Actors read the property, including changes made outside the actor API."""
    shape = pv.PointSpriteShape(shape) if as_enum else shape
    actor = pv.Actor()
    actor.set_point_sprite_shape(shape)
    assert actor.prop.GetPoint2DShape() == getattr(actor.prop.Point2DShapeType, native_name)
    assert actor.prop.point_shape == shape
    assert not actor._shader_replacements
    actor.prop.point_shape = 'square'
    assert actor.point_sprite_shape == 'square'
    actor.prop.point_shape = shape
    assert actor.point_sprite_shape == shape
    actor.clear_point_sprite_shape()
    assert actor.point_sprite_shape == 'square'


@pytest.mark.parametrize('shape', list(pv.PointSpriteShape))
def test_native_shape_copy(shape):
    """Real C++ DeepCopy preserves shape independently of the original actor."""
    actor = pv.Actor()
    actor.set_point_sprite_shape(shape)
    restored = pv.Actor(prop=actor.prop.copy())
    expected_shape = actor.prop.point_shape
    assert expected_shape == (shape if _HAS_NATIVE_POINT_SHAPES else 'square')
    actor.clear_point_sprite_shape()
    assert restored.point_sprite_shape == expected_shape


@pytest.mark.parametrize('style', ['surface', 'wireframe', 'points'])
def test_theme_circle_reaches_rendered_vertex_cells(style):
    """An omitted shape renders circles from the plotter's own theme."""
    if style != 'points' and not _HAS_NATIVE_POINT_SHAPES:
        pytest.skip('Requires native point-shape rendering outside points representation')
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


@pytest.mark.usefixtures('native_property_api')
def test_explicit_spheres_override_theme_circle():
    """The theme's flat shape does not disable an explicit sphere request."""
    theme = pv.themes.Theme()
    theme.point_shape = 'circle'
    pl = pv.Plotter(theme=theme)
    actor = pl.add_mesh(pv.PolyData(np.array([[0.0, 0.0, 0.0]])), render_points_as_spheres=True)
    assert actor.prop.render_points_as_spheres
    assert actor.point_sprite_shape == 'circle'
    pl.close()


@pytest.mark.usefixtures('native_property_api')
def test_property_uses_its_own_theme():
    """A property created without add_mesh still resolves the theme default."""
    theme = pv.themes.Theme()
    theme.point_shape = 'diamond'
    assert pv.Property(theme=theme).point_shape == theme.point_shape


@pytest.mark.usefixtures('native_property_api')
def test_invalid_native_shape_preserves_the_previous_shape():
    """Invalid input cannot silently clear the selected shape."""
    prop = pv.Property()
    prop.point_shape = 'circle'
    with pytest.raises(ValueError, match='Invalid point sprite shape'):
        prop.point_shape = 'pentagon'
    assert prop.point_shape == 'circle'


@pytest.mark.usefixtures('native_property_api')
def test_unknown_native_shape(monkeypatch):
    """An unrecognized backend value is not silently reported as square."""
    prop = pv.Property()
    # mocked: the binding cannot construct an unknown C++ enum value.
    monkeypatch.setattr(pv.Property, 'GetPoint2DShape', lambda _self: 99)
    with pytest.raises(ValueError, match='Unknown native point shape'):
        _ = prop.point_shape


def test_property_shape_without_native_support(monkeypatch):
    """Unsupported setters refuse valid shapes and validate invalid inputs."""
    # mocked: exercise the unsupported-backend contract on native backends too.
    monkeypatch.setattr('pyvista.plotting._property._HAS_NATIVE_POINT_SHAPES', False)
    prop = pv.Property()
    assert prop.point_shape == 'square'
    with pytest.raises(pv.VTKVersionError, match='does not support native point shapes'):
        prop.point_shape = 'circle'
    with pytest.raises(ValueError, match='Invalid point sprite shape'):
        prop.point_shape = 'pentagon'
    assert prop.point_shape == 'square'
