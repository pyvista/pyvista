from __future__ import annotations

import numpy as np
import pytest

import pyvista as pv

# Valid range of every Property attribute which restricts its value.
PROPERTY_RANGES = {
    'ambient': (0.0, 1.0),
    'anisotropy': (0.0, 1.0),
    'anisotropy_rotation': (0.0, 1.0),
    'diffuse': (0.0, 1.0),
    'edge_opacity': (0.0, 1.0),
    'index_of_refraction': (1.0, np.inf),
    'line_width': (0.0, np.inf),
    'metallic': (0.0, 1.0),
    'opacity': (0.0, 1.0),
    'point_size': (0.0, np.inf),
    'roughness': (0.0, 1.0),
    'specular': (0.0, 1.0),
    'specular_power': (0.0, 128.0),
}


@pytest.fixture
def prop():
    return pv.Property()


@pytest.mark.parametrize(('name', 'rng'), PROPERTY_RANGES.items(), ids=PROPERTY_RANGES)
def test_property_range_is_validated(prop, name, rng):
    lower, upper = rng
    setattr(prop, name, lower)
    assert getattr(prop, name) == lower
    with pytest.raises(ValueError, match=f'{name} values must all be greater than or equal'):
        setattr(prop, name, lower - 1.0)

    if np.isfinite(upper):
        setattr(prop, name, upper)
        assert getattr(prop, name) == upper
        with pytest.raises(ValueError, match=f'{name} values must all be less than or equal'):
            setattr(prop, name, upper + 1.0)

    else:
        setattr(prop, name, 1e6)
        assert getattr(prop, name) == 1e6
        with pytest.raises(ValueError, match=f'{name} values must all be less than inf'):
            setattr(prop, name, np.inf)


@pytest.mark.parametrize(('name', 'rng'), PROPERTY_RANGES.items(), ids=PROPERTY_RANGES)
def test_property_range_is_documented(name, rng):
    lower, upper = rng
    upper_bound = 'inf)' if np.isinf(upper) else f'{upper}]'
    expected = f'Property has range ``[{lower}, {upper_bound}``.'
    assert expected in getattr(pv.Property, name).__doc__


def test_property_init():
    prop = pv.Property()

    # copy but equal
    assert prop._theme is not pv.global_theme
    assert prop._theme == pv.global_theme


def test_property_style(prop):
    style = 'Surface'
    prop.style = style
    assert prop.style == style


def test_property_representation(prop):
    from pyvista.plotting.opts import RepresentationType

    # default
    assert prop.representation == RepresentationType.SURFACE
    assert prop.style == 'Surface'

    # set via enum, string (case insensitive), and int
    prop.representation = RepresentationType.WIREFRAME
    assert prop.representation == RepresentationType.WIREFRAME
    assert prop.style == 'Wireframe'

    prop.representation = 'POINTS'
    assert prop.representation == RepresentationType.POINTS
    assert prop.style == 'Points'

    prop.representation = 2
    assert prop.representation == RepresentationType.SURFACE

    # invalid values raise
    with pytest.raises(ValueError):  # noqa: PT011
        prop.representation = 'invalid'
    with pytest.raises(ValueError):  # noqa: PT011
        prop.representation = 99


def test_property_style_accepts_representation_type(prop):
    # Regression test for https://github.com/pyvista/pyvista/issues/8168
    # `style` setter previously only accepted strings.
    from pyvista.plotting.opts import RepresentationType

    prop.style = RepresentationType.WIREFRAME
    assert prop.style == 'Wireframe'
    assert prop.representation == RepresentationType.WIREFRAME
    # `style` getter still returns a string for backwards compatibility
    assert isinstance(prop.style, str)


def test_property_representation_wireframe_color(prop):
    # setting wireframe applies the theme outline color when no color is set
    prop.representation = 'wireframe'
    assert prop.color == pv.Color(prop._theme.outline_color)


def test_property_edge_color(prop):
    prop.edge_color = 'b'
    assert prop.edge_color.float_rgb == (0, 0, 1)


def test_property_opacity(prop):
    opacity = 0.5
    prop.opacity = opacity
    assert prop.opacity == opacity


def test_property_edge_opacity(prop):
    edge_opacity = 0.5
    prop.edge_opacity = edge_opacity
    assert prop.edge_opacity == edge_opacity


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


def test_property_diffuse(prop):
    value = 0.5
    prop.diffuse = value
    assert prop.diffuse == value


def test_property_specular(prop):
    value = 0.5
    prop.specular = value
    assert prop.specular == value


def test_property_specular_power(prop):
    value = 0.5
    prop.specular_power = value
    assert prop.specular_power == value


def test_property_metallic(prop):
    value = 0.1
    prop.metallic = value
    assert prop.metallic == value


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


@pytest.mark.parametrize(
    ('interpolation', 'lit_by_environment'),
    [('pbr', True), ('phong', False)],
)
def test_property_plot_environment_texture(prop, interpolation, lit_by_environment):
    """Only physically based rendering is lit by the skybox environment texture."""
    prop.interpolation = interpolation
    captured = {}

    def capture(pl):
        captured['image_based_lighting'] = bool(pl.renderer.GetUseImageBasedLighting())
        captured['environment_texture'] = pl.renderer.GetEnvironmentTexture() is not None

    prop.plot(before_close_callback=capture)

    assert captured['image_based_lighting'] is lit_by_environment
    assert captured['environment_texture'] is lit_by_environment


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


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        (True, 'back'),
        ('b', 'back'),
        ('back', 'back'),
        ('backface', 'back'),
        ('f', 'front'),
        ('front', 'front'),
        ('frontface', 'front'),
        (False, 'none'),
        ('none', 'none'),
        ('BackFace', 'back'),
    ],
)
def test_property_culling(prop, value, expected):
    prop.culling = value
    assert prop.culling == expected

    assert pv.Property(culling=value).culling == expected

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


def test_property_anisotropy_rotation(prop):
    assert prop.anisotropy_rotation == 0.0
    prop.anisotropy_rotation = 0.25
    assert prop.anisotropy_rotation == 0.25


def test_property_index_of_refraction(prop):
    assert prop.index_of_refraction == 1.5
    prop.index_of_refraction = 2.0
    assert prop.index_of_refraction == 2.0
