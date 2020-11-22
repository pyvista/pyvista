import math

import pytest
import vtk

import pyvista
from pyvista.plotting import system_supports_plotting
# TODO: do we need OFF_SCREEN stuff? And as per tests/test_renderer.py or tests/plotting/test_plotting.py?
# TODO: do we actually need the @skip_no_plotting decorator when there's no plotter?

skip_no_plotting = pytest.mark.skipif(not system_supports_plotting(),
                                      reason="Test requires system to support plotting")

# TODO: invalid cases, once checks are in place

@skip_no_plotting
def test_init():
    position = (1, 1, 1)
    color = (0.5, 0.5, 0.5)
    light_type = 'headlight'
    light = pyvista.Light(position=position, color=color, light_type=light_type)
    assert isinstance(light, pyvista.Light)
    assert light.position == position
    assert light.ambient_color == color
    assert light.diffuse_color == color
    assert light.specular_color == color
    assert light.light_type == light.HEADLIGHT

    # check repr too
    assert repr(light) is not None


@skip_no_plotting
def test_colors():
    light = pyvista.Light()

    color = (0, 1, 0)
    light.diffuse_color = color
    assert light.diffuse_color == color
    color = (0, 0, 1)
    light.specular_color = color
    assert light.specular_color == color
    color = (1, 0, 0)
    light.ambient_color = color
    assert light.ambient_color == color

    old_color, color = color, (1, 1, 0)
    light.set_color(color)
    assert light.diffuse_color == light.specular_color == color
    assert light.ambient_color == old_color


@skip_no_plotting
def test_positioning():
    light = pyvista.Light()

    position = (1, 1, 1)
    light.position = position
    assert light.position == position
    # with no transformation matrix this is also the world position
    assert light.world_position == position

    focal_point = (2, 2, 2)
    light.focal_point = focal_point
    assert light.focal_point == focal_point
    # with no transformation matrix this is also the world focal point
    assert light.world_focal_point == focal_point

    elev, azim = (45, 30)
    expected_position = (0.5 / math.sqrt(2),
                         1 / math.sqrt(2),
                         math.sqrt(3) / (2 * math.sqrt(2)))  # TODO: fix this style
    light.positional = True
    light.set_direction_angle(elev, azim)
    assert not light.positional
    assert light.focal_point == (0, 0, 0)
    assert all(math.isclose(coord_have, coord_expect) for coord_have, coord_expect
               in zip(light.position, expected_position))  # TODO: fix this style


@skip_no_plotting
def test_intensity():
    light = pyvista.Light()

    intensity = 0.5
    light.intensity = intensity
    assert light.intensity == intensity


@skip_no_plotting
def test_switch_state():
    light = pyvista.Light()

    light.switch_on()
    assert light.is_on
    light.switch_off()
    assert not light.is_on
    light.switch()
    assert light.is_on
    light.is_on = False
    assert not light.is_on


@skip_no_plotting
def test_positional():
    light = pyvista.Light()

    # default is directional light
    assert not light.positional
    light.positional_on()
    assert light.positional
    light.positional_off()
    assert not light.positional
    light.positional = True
    assert light.positional


@skip_no_plotting
def test_shape():
    light = pyvista.Light()

    exponent = 1.5
    light.exponent = exponent
    assert light.exponent == exponent

    cone_angle = 45
    light.cone_angle = cone_angle
    assert light.cone_angle == cone_angle

    attenuation_values = (3, 2, 1)
    light.attenuation_values = attenuation_values
    assert light.attenuation_values == attenuation_values

    shadow_attenuation = 0.5
    light.shadow_attenuation = shadow_attenuation
    assert light.shadow_attenuation == shadow_attenuation


@skip_no_plotting
@pytest.mark.parametrize(
    'int_code,enum_code',
    [
        (1, pyvista.Light.HEADLIGHT),
        (2, pyvista.Light.CAMERA_LIGHT),
        (3, pyvista.Light.SCENE_LIGHT),
    ]
)
def test_type_properties(int_code, enum_code):
    light = pyvista.Light()

    # test that the int and enum codes match up
    assert int_code == enum_code

    # test that both codes work
    light.light_type = int_code
    assert light.light_type == int_code
    light.light_type = enum_code
    assert light.light_type == enum_code


@skip_no_plotting
def test_type_setters():
    light = pyvista.Light()

    light.set_headlight()
    assert light.is_headlight
    light.set_camera_light()
    assert light.is_camera_light
    light.set_scene_light()
    assert light.is_scene_light


@skip_no_plotting
def test_type_invalid():
    with pytest.raises(TypeError):
        light = pyvista.Light(light_type=['invalid'])
    with pytest.raises(ValueError):
        light = pyvista.Light(light_type='invalid')

    light = pyvista.Light()

    with pytest.raises(TypeError):
        light.light_type = ['invalid']


@skip_no_plotting
def test_from_vtk():
    vtk_light = vtk.vtkLight()

    # pyvista attr -- value -- vtk name triples:
    configuration = [
        ('light_type', pyvista.Light.CAMERA_LIGHT, 'SetLightType'),  # resets transformation!
        ('position', (1, 1, 1), 'SetPosition'),
        ('focal_point', (2, 2, 2), 'SetFocalPoint'),
        ('ambient_color', (1, 0, 0), 'SetAmbientColor'),
        ('diffuse_color', (0, 1, 0), 'SetDiffuseColor'),
        ('specular_color', (0, 0, 1), 'SetSpecularColor'),
        ('intensity', 0.5, 'SetIntensity'),
        ('is_on', False, 'SetSwitch'),
        ('positional', True, 'SetPositional'),
        ('exponent', 1.5, 'SetExponent'),
        ('cone_angle', 45, 'SetConeAngle'),
        ('attenuation_values', (3, 2, 1), 'SetAttenuationValues'),
        ('shadow_attenuation', 0.5, 'SetShadowAttenuation'),
    ]

    # set the vtk light
    for _, value, vtkname in configuration:
        vtk_setter = getattr(vtk_light, vtkname)
        vtk_setter(value)
    light = pyvista.Light.from_vtk(vtk_light)
    for pvname, value, _ in configuration:
        assert getattr(light, pvname) == value

    # invalid case
    with pytest.raises(TypeError):
        pyvista.Light.from_vtk('invalid')
    with pytest.raises(TypeError):
        pyvista.Light('invalid')
