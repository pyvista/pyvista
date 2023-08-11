"""
Tests for text objects
"""
import pyvista


def test_text():
    text = pyvista.Text('text')
    prop = text.prop
    assert isinstance(prop, pyvista.TextProperty)


def test_text_prop():
    prop = pyvista.TextProperty()
    prop.color = "b"
    prop.opacity = 0.5
    prop.background_color = "b"
    prop.background_opacity = 0.5
    prop.frame is True
    prop.frame_color = "b"
    prop.frame_width = 10.0
    assert prop.color == "b"
    assert prop.opacity == 0.5
    assert prop.background_color == "b"
    assert prop.background_opacity == 0.5
    assert prop.frame is True
    assert prop.frame_color == "b"
    assert prop.frame_width == 10.0
