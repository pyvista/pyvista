import pytest

import pyvista as pv


@pytest.fixture()
def prop():
    return pv.Property()


def test_property_init():
    prop = pv.Property()

    # copy but equal
    assert prop._theme is not pv.global_theme
    assert prop._theme == pv.global_theme


def test_property_style(prop):
    style = 'Surface'
    prop.style = style
    assert prop.style == style


def test_property_edge_color(prop):
    prop.edge_color = 'b'
    assert prop.edge_color.float_rgb == (0, 0, 1)


def test_property_opacity(prop):
    opacity = 0.5
    prop.opacity = opacity
    assert prop.opacity == opacity


def test_property_show_edges(prop):
    value = False
    prop.show_edges = value
    assert prop.show_edges == value


def test_property_lighting(prop):
    value = False
    prop.lighting = value
    assert prop.lighting == value
