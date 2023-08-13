"""
Tests for text objects
"""
import numpy as np
import pytest

import pyvista as pv


@pytest.fixture()
def corner_annotation():
    return pv.CornerAnnotation()


def test_corner_annotation_text(corner_annotation):
    corner_annotation.set_text(0, 'text')
    assert corner_annotation.get_text(0) == 'text'


def test_corner_annotation_prop(corner_annotation):
    prop = corner_annotation.prop
    assert isinstance(prop, pv.TextProperty)


@pytest.fixture()
def text():
    return pv.Text()


def test_text_input(text):
    text.input = 'input'
    assert text.input == 'input'


def test_text_prop(text):
    prop = text.prop
    assert isinstance(prop, pv.TextProperty)


def test_text_position(text):
    position = np.random.random(2)
    text.position = position
    assert np.all(text.position == position)


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


def test_property_font_family(prop):
    prop.font_family = 'arial'
    assert prop.font_family == 'arial'


def test_property_font_size(prop):
    assert isinstance(prop.font_size, int)
    value = 10
    prop.font_size = value
    assert prop.font_size == value
