"""Tests for text objects"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pytest

import pyvista as pv
from pyvista.plotting.text import _TEXT_POSITIONS


@pytest.fixture
def corner_annotation():
    return pv.CornerAnnotation(0, 'text')


def test_corner_annotation_text(corner_annotation):
    corner_annotation.set_text(1, 'text1')
    assert corner_annotation.get_text(1) == 'text1'


def test_corner_annotation_prop(corner_annotation):
    prop = corner_annotation.prop
    assert isinstance(prop, pv.TextProperty)


def test_corner_annotation_name():
    corner = pv.CornerAnnotation('top', 'abc', name='corner')
    assert corner.name == 'corner'


@pytest.fixture
def text():
    return pv.Text()


def test_text_input(text):
    text.input = 'input'
    assert text.input == 'input'


def test_text_prop(text):
    prop = text.prop
    assert isinstance(prop, pv.TextProperty)


def test_text_position(text):
    position = np.random.default_rng().random(2)
    text.position = position
    assert np.all(text.position == position)


def test_text_name():
    text = pv.Text('abc', name='text')
    assert text.name == 'text'


def test_label():
    label = pv.Label('text', (1, 2, 3), size=42, prop=pv.Property())

    assert label.input == 'text'
    label.input = 'new'
    assert label.input == 'new'

    assert label.position == (1, 2, 3)
    label.position = (4, 5, 6)
    assert label.position == (4, 5, 6)

    assert label.size == 42
    label.size = 99
    assert label.size == 99


def test_label_name():
    label = pv.Label('abc', name='label')
    assert label.name == 'label'


def test_label_prop3d():
    position = (1.0, 2.0, 3.0)
    label = pv.Label(position=position)
    bounds = (1.0, 1.0, 2.0, 2.0, 3.0, 3.0)
    assert label.bounds == bounds
    assert label.center == position
    assert label.length == 0.0

    # Test correct bounds with more complex transformations
    # Add offset along x-axis
    offset = 100
    label.relative_position = (offset, 0, 0)
    # Rotate about z-axis
    label.orientation = (0, 0, 90)
    # Expect offset to be applied along y-axis (due to the rotation)
    bounds = (
        position[0],
        position[0],
        position[1] + offset,
        position[1] + offset,
        position[2],
        position[2],
    )
    assert np.allclose(label.bounds, bounds)


def test_label_relative_position():
    label = pv.Label()
    position = (1, 2, 3)
    label.position = position
    assert label.position == position
    assert label._prop3d.position == position
    assert label._label_position == position

    relative_position = np.array(position) * -1
    label.relative_position = relative_position
    assert label.position == position
    assert label._prop3d.position == position
    assert label._label_position == tuple((position + relative_position).tolist())


@pytest.fixture
def prop():
    return pv.TextProperty()


def test_property_init():
    prop = pv.TextProperty()

    # copy but equal
    assert prop._theme is not pv.global_theme
    assert prop._theme == pv.global_theme


def test_property_color(prop):
    prop.color = 'b'
    assert prop.color == 'b'


def test_property_opacity(prop):
    opacity = 0.5
    prop.opacity = opacity
    assert prop.opacity == opacity
    with pytest.raises(ValueError):  # noqa: PT011
        prop.opacity = 2


def test_property_background_color(prop):
    prop.background_color = 'b'
    assert prop.background_color.float_rgb == (0, 0, 1)


def test_property_background_opacity(prop):
    background_opacity = 0.5
    prop.background_opacity = background_opacity
    assert prop.background_opacity == background_opacity
    with pytest.raises(ValueError):  # noqa: PT011
        prop.background_opacity = 2


def test_property_show_frame(prop):
    value = False
    prop.show_frame = value
    assert prop.show_frame == value


def test_property_frame_color(prop):
    prop.frame_color = 'b'
    assert prop.frame_color.float_rgb == (0, 0, 1)


def test_property_frame_width(prop):
    assert isinstance(prop.frame_width, int)
    value = 10
    prop.frame_width = value
    assert prop.frame_width == value


def test_property_font_family(prop):
    prop.font_family = 'arial'
    assert prop.font_family == 'arial'


def test_property_font_size(prop):
    assert isinstance(prop.font_size, int)
    value = 10
    prop.font_size = value
    assert prop.font_size == value


def test_property_enable_shadow(prop):
    prop.enable_shadow()
    assert prop.GetShadow() == 1


def test_property_orientation(prop):
    prop.orientation = 90.0
    assert prop.orientation == 90.0


def test_property_set_font_file(prop):
    font_file = str(Path(__file__).parent / 'fonts/Mplus2-Regular.ttf')
    prop.set_font_file(font_file)
    with pytest.raises(FileNotFoundError):
        prop.set_font_file('foo.ttf')


@pytest.mark.parametrize(
    'justification',
    [('left', 'left'), ('center', 'centered'), ('right', 'right')],
)
def test_property_justification_horizontal(prop, justification):
    prop.justification_horizontal = justification[0]
    assert prop.GetJustificationAsString().lower() == justification[1]
    assert prop.justification_horizontal == justification[0]
    prop = pv.TextProperty(justification_horizontal=justification[0])
    assert prop.GetJustificationAsString().lower() == justification[1]
    assert prop.justification_horizontal == justification[0]


@pytest.mark.parametrize(
    'justification',
    [('bottom', 'bottom'), ('center', 'centered'), ('top', 'top')],
)
def test_property_justification_vertical(prop, justification):
    prop.justification_vertical = justification[0]
    assert prop.GetVerticalJustificationAsString().lower() == justification[1]
    assert prop.justification_vertical == justification[0]
    prop = pv.TextProperty(justification_vertical=justification[0])
    assert prop.GetVerticalJustificationAsString().lower() == justification[1]
    assert prop.justification_vertical == justification[0]


def test_property_justification_invalid(prop):
    with pytest.raises(ValueError):  # noqa: PT011
        prop.justification_horizontal = 'invalid'
    with pytest.raises(ValueError):  # noqa: PT011
        prop.justification_vertical = 'invalid'


@pytest.mark.parametrize('italic', [True, False])
def test_property_italic(prop, italic):
    prop.italic = italic
    assert prop.GetItalic() == italic
    assert prop.italic == italic


@pytest.mark.parametrize('bold', [True, False])
def test_property_bold(prop, bold):
    prop.bold = bold
    assert prop.GetBold() == bold
    assert prop.bold == bold


@pytest.mark.parametrize('italic', [True, False])
@pytest.mark.parametrize('bold', [True, False])
def test_property_shallow_copy(prop, italic, bold):
    prop.italic = italic
    prop.bold = bold
    text_prop = pv.TextProperty()
    text_prop.shallow_copy(prop)
    assert text_prop.bold == prop.bold


@pytest.mark.parametrize('position', list(_TEXT_POSITIONS))
def test_add_text_actor_named_position(position):
    """Test that each named position is placed within the viewport it is drawn in."""
    pl = pv.Plotter()
    actor = pl._add_text_actor('text', position=position)

    assert isinstance(actor, pv.Text)
    # A named position is placed as a fraction of the viewport, so that the text is
    # drawn in the same part of it whatever size the window is
    x, y = actor.position
    assert 0 <= x <= 1
    assert 0 <= y <= 1
    coordinate = actor.GetActualPositionCoordinate()
    assert coordinate.GetCoordinateSystemAsString() == 'Normalized Viewport'

    # The text is anchored to the part of the viewport it is placed in, so that it
    # stays there whatever size it ends up being drawn at
    horizontal, vertical = actor.prop.justification_horizontal, actor.prop.justification_vertical
    assert horizontal == ('center' if x == 0.5 else 'left' if x < 0.5 else 'right')
    assert vertical == ('center' if y == 0.5 else 'bottom' if y < 0.5 else 'top')
    pl.close()


def test_add_text_actor_font_size_matches_add_text():
    """Test that a font size means the same to this as it does to `add_text`."""
    pl = pv.Plotter()
    drawn = pl.add_text('text', position=(10, 10), font_size=24)
    named = pl._add_text_actor('text', position='upper_left', font_size=24)
    assert named.prop.font_size == drawn.prop.font_size

    # The theme says how big the text is by default
    assert pl._add_text_actor('text').prop.font_size == pl.theme.font.size * 2
    pl.close()


def test_add_text_actor_coordinate_position():
    """Test that a coordinate is used as it is, in pixels of the viewport unless told
    to read it as a fraction of the size of the viewport instead.
    """
    pl = pv.Plotter()
    for viewport, expected in [(False, 'Viewport'), (True, 'Normalized Viewport')]:
        actor = pl._add_text_actor('text', position=(10, 20), viewport=viewport)
        assert tuple(actor.position) == (10, 20)
        assert actor.GetActualPositionCoordinate().GetCoordinateSystemAsString() == expected
    pl.close()


def test_add_text_actor_raises():
    pl = pv.Plotter()
    match = "Position 'middle' is not a coordinate or one of 'lower_left', "
    with pytest.raises(ValueError, match=re.escape(match)):
        pl._add_text_actor('text', position='middle')
    pl.close()


def test_add_text_actor_follows_the_theme_of_the_plotter():
    """Test that text is drawn in the color and font its own plotter asks for."""
    theme = pv.themes.Theme()
    theme.font.color = 'red'
    theme.font.family = 'courier'
    pl = pv.Plotter(theme=theme)

    # A plotter with a theme of its own, a dark one in particular, would otherwise
    # draw text in the color of the global theme, which its background may well be
    actor = pl._add_text_actor('text')
    assert actor.prop.color == pv.Color('red')
    assert actor.prop.font_family == 'courier'

    # What the caller asks for is used instead
    actor = pl._add_text_actor('text', color='blue', font='times')
    assert actor.prop.color == pv.Color('blue')
    assert actor.prop.font_family == 'times'
    pl.close()
