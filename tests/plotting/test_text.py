"""
Tests for text objects
"""
import pytest

import pyvista as pv


def test_text():
    text = pv.Text('text')
    prop = text.prop
    assert isinstance(prop, pv.TextProperty)


@pytest.fixture()
def prop():
    return pv.TextProperty()


def test_property_init():
    prop = pv.TextProperty()

    # copy but equal
    assert prop._theme is not pv.global_theme
    assert prop._theme == pv.global_theme


def test_property_color(prop):
    prop.color = "b"
    assert prop.color == "b"


def test_property_opacity(prop):
    opacity = 0.5
    prop.opacity = opacity
    assert prop.opacity == opacity
    with pytest.raises(ValueError):
        prop.opacity = 2


def test_property_background_color(prop):
    prop.background_color = 'b'
    assert prop.background_color.float_rgb == (0, 0, 1)


def test_property_background_opacity(prop):
    background_opacity = 0.5
    prop.background_opacity = background_opacity
    assert prop.background_opacity == background_opacity
    with pytest.raises(ValueError):
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
