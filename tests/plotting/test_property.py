from __future__ import annotations

import pytest

import pyvista as pv
from pyvista.plotting._property import _check_range


@pytest.fixture
def prop():
    return pv.Property()


def test_check_range():
    with pytest.raises(ValueError, match='outside the acceptable'):
        _check_range(-1, (0, 1), 'parm')
    with pytest.raises(ValueError, match='outside the acceptable'):
        _check_range(2, (0, 1), 'parm')
    assert _check_range(0, (0, 1), 'parm') is None


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
    with pytest.raises(ValueError):  # noqa: PT011
        prop.opacity = 2


@pytest.mark.needs_vtk_version(9, 3, reason='Functions not implemented before 9.3.X')
def test_property_edge_opacity(prop):
    edge_opacity = 0.5
    prop.edge_opacity = edge_opacity
    assert prop.edge_opacity == edge_opacity
    with pytest.raises(ValueError):  # noqa: PT011
        prop.edge_opacity = 2


def test_property_show_edges(prop):
    value = False
    prop.show_edges = value
    assert prop.show_edges == value


def test_property_lighting(prop):
    value = False
    prop.lighting = value
    assert prop.lighting == value


def test_property_ambient(prop):
    value = 0.45
    prop.ambient = value
    assert prop.ambient == value
    with pytest.raises(ValueError):  # noqa: PT011
        prop.ambient = -1


def test_property_diffuse(prop):
    value = 0.5
    prop.diffuse = value
    assert prop.diffuse == value
    with pytest.raises(ValueError):  # noqa: PT011
        prop.diffuse = 2


def test_property_specular(prop):
    value = 0.5
    prop.specular = value
    assert prop.specular == value
    with pytest.raises(ValueError):  # noqa: PT011
        prop.specular = 2


def test_property_specular_power(prop):
    value = 0.5
    prop.specular_power = value
    assert prop.specular_power == value
    with pytest.raises(ValueError):  # noqa: PT011
        prop.specular = 200


def test_property_metallic(prop):
    value = 0.1
    prop.metallic = value
    assert prop.metallic == value
    with pytest.raises(ValueError):  # noqa: PT011
        prop.metallic = -1


def test_property_roughness(prop):
    value = 0.1
    prop.roughness = value
    assert prop.roughness == value


def test_property_interpolation(prop):
    value = 'Gouraud'
    prop.interpolation = value
    assert prop.interpolation == pv.opts.InterpolationType.from_any(value)

    with pytest.raises(ValueError, match='InterpolationType has no value matching'):
        prop.interpolation = 'foo'


def test_property_render_points_as_spheres(prop):
    value = True
    prop.render_points_as_spheres = value
    assert prop.render_points_as_spheres is value


def test_property_render_lines_as_tubes(prop):
    value = True
    prop.render_lines_as_tubes = value
    assert prop.render_lines_as_tubes is value


def test_property_point_size(prop):
    value = 10.0
    prop.point_size = value
    assert prop.point_size == value


def test_property_line_width(prop):
    assert isinstance(prop.line_width, float)
    value = 10.0
    prop.line_width = value
    assert prop.line_width == value


@pytest.mark.parametrize('value', ['back', 'front', 'none'])
def test_property_culling(prop, value):
    prop.culling = value
    assert prop.culling == value

    with pytest.raises(ValueError, match='Invalid culling'):
        prop.culling = 'foo'


def test_property_diffuse_color(prop):
    prop.diffuse_color = 'b'
    assert prop.diffuse_color.float_rgb == (0, 0, 1)


def test_property_ambient_color(prop):
    prop.ambient_color = 'b'
    assert prop.ambient_color.float_rgb == (0, 0, 1)


def test_property_specular_color(prop):
    prop.specular_color = 'b'
    assert prop.specular_color.float_rgb == (0, 0, 1)


def test_property_anisotropy(prop):
    value = 0.1
    assert isinstance(prop.anisotropy, float)
    prop.anisotropy = value
    assert prop.anisotropy == value
