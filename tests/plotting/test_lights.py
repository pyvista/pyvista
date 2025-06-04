from __future__ import annotations

import numpy as np
import pytest
import vtk

import pyvista as pv

# pyvista attr -- value -- vtk name triples:
configuration = [
    ('light_type', pv.Light.CAMERA_LIGHT, 'SetLightType'),  # resets transformation!
    ('position', (1, 1, 1), 'SetPosition'),
    ('focal_point', (2, 2, 2), 'SetFocalPoint'),
    ('ambient_color', (1.0, 0.0, 0.0), 'SetAmbientColor'),
    ('diffuse_color', (0.0, 1.0, 0.0), 'SetDiffuseColor'),
    ('specular_color', (0.0, 0.0, 1.0), 'SetSpecularColor'),
    ('intensity', 0.5, 'SetIntensity'),
    ('on', False, 'SetSwitch'),
    ('positional', True, 'SetPositional'),
    ('exponent', 1.5, 'SetExponent'),
    ('cone_angle', 45, 'SetConeAngle'),
    ('attenuation_values', (3, 2, 1), 'SetAttenuationValues'),
    ('transform_matrix', np.arange(4 * 4).reshape(4, 4), 'SetTransformMatrix'),
    ('shadow_attenuation', 0.5, 'SetShadowAttenuation'),
]


def test_init():
    position = (1, 1, 1)
    focal_point = (2, 2, 2)
    color = (0.5, 0.5, 0.5)
    light_type = 'headlight'
    cone_angle = 15
    intensity = 2
    exponent = 1.5
    positional = True
    show_actor = False
    shadow_attenuation = 0.5
    light = pv.Light(
        position=position,
        focal_point=focal_point,
        color=color,
        light_type=light_type,
        cone_angle=cone_angle,
        intensity=intensity,
        exponent=exponent,
        show_actor=show_actor,
        positional=positional,
        shadow_attenuation=shadow_attenuation,
    )
    assert isinstance(light, pv.Light)
    assert light.position == position
    assert light.focal_point == focal_point
    assert light.ambient_color == color
    assert light.diffuse_color == color
    assert light.specular_color == color
    assert light.light_type == light.HEADLIGHT
    assert light.cone_angle == cone_angle
    assert light.intensity == intensity
    assert light.exponent == exponent
    assert light.positional == positional
    assert light.actor.GetVisibility() == show_actor
    assert light.shadow_attenuation == shadow_attenuation

    # check repr too
    assert repr(light) is not None


def test_eq():
    light = pv.Light()
    other = pv.Light()
    for light_now in light, other:
        for name, value, _ in configuration:
            setattr(light_now, name, value)

    assert light == other

    # check that changing anything will break equality
    for name, value, _ in configuration:
        original_value = getattr(other, name)
        restore_transform = False
        if value is pv.Light.CAMERA_LIGHT:
            changed_value = pv.Light.HEADLIGHT
            restore_transform = True
        elif isinstance(value, bool):
            changed_value = not value
        elif isinstance(value, (int, float)):
            changed_value = 0
        elif isinstance(value, tuple):
            changed_value = (0.5, 0.5, 0.5)
        else:
            # transform_matrix; value is an ndarray
            changed_value = -value
        setattr(other, name, changed_value)
        assert light != other
        setattr(other, name, original_value)
        if restore_transform:
            # setting the light type cleared the transform
            other.transform_matrix = light.transform_matrix

    # sanity check that we managed to restore the original state
    assert light == other

    # check None vs transform_matrix case
    other.transform_matrix = None
    assert light != other


def test_copy():
    light = pv.Light()
    for name, value, _ in configuration:
        setattr(light, name, value)

    deep = light.copy()
    assert deep == light
    assert deep.transform_matrix is not light.transform_matrix
    shallow = light.copy(deep=False)
    assert shallow == light
    assert shallow.transform_matrix is light.transform_matrix


def test_colors():
    light = pv.Light()

    color = (0.0, 1.0, 0.0)
    light.diffuse_color = color
    assert light.diffuse_color == color
    color = (0.0, 0.0, 1.0)
    light.specular_color = color
    assert light.specular_color == color
    color = (1.0, 0.0, 0.0)
    light.ambient_color = color
    assert light.ambient_color == color

    # test whether strings raise but don't test the result
    for valid in 'white', 'r', '#c0ffee':
        light.diffuse_color = valid
        light.specular_color = valid
        light.ambient_color = valid
    with pytest.raises(ValueError):  # noqa: PT011
        light.diffuse_color = 'invalid'
    with pytest.raises(ValueError):  # noqa: PT011
        light.specular_color = 'invalid'
    with pytest.raises(ValueError):  # noqa: PT011
        light.ambient_color = 'invalid'


def test_positioning():
    light = pv.Light()

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

    elev, azim = (30, 60)
    expected_position = (np.sqrt(3) / 2 * 1 / 2, np.sqrt(3) / 2 * np.sqrt(3) / 2, 1 / 2)
    light.positional = True
    light.set_direction_angle(elev, azim)
    assert not light.positional
    assert light.focal_point == (0, 0, 0)
    assert np.allclose(light.position, expected_position)

    with pytest.raises(AttributeError):
        light.world_position = position
    with pytest.raises(AttributeError):
        light.world_focal_point = focal_point


def test_transforms():
    position = (1, 2, 3)
    focal_point = (4, 5, 6)
    light = pv.Light(position=position)
    light.focal_point = focal_point

    trans_array = np.arange(4 * 4).reshape(4, 4)
    trans_matrix = pv.vtkmatrix_from_array(trans_array)

    assert light.transform_matrix is None
    light.transform_matrix = trans_array
    assert isinstance(light.transform_matrix, vtk.vtkMatrix4x4)
    array = pv.array_from_vtkmatrix(light.transform_matrix)
    assert np.array_equal(array, trans_array)
    light.transform_matrix = trans_matrix
    matrix = light.transform_matrix
    assert all(
        matrix.GetElement(i, j) == trans_matrix.GetElement(i, j)
        for i in range(4)
        for j in range(4)
    )

    linear_trans = trans_array[:-1, :-1]
    shift = trans_array[:-1, -1]
    assert light.position == position
    assert np.allclose(light.world_position, linear_trans @ position + shift)
    assert light.focal_point == focal_point
    assert np.allclose(light.world_focal_point, linear_trans @ focal_point + shift)

    with pytest.raises(TypeError, match='Input transform must be one of'):
        light.transform_matrix = 'invalid'


def test_intensity():
    light = pv.Light()

    intensity = 0.5
    light.intensity = intensity
    assert light.intensity == intensity


def test_switch_state():
    light = pv.Light()

    light.switch_on()
    assert light.on
    light.switch_off()
    assert not light.on
    light.on = False
    assert not light.on


def test_positional():
    light = pv.Light()

    # default is directional light
    assert not light.positional
    light.positional = True
    assert light.positional
    light.positional = False
    assert not light.positional


def test_shape():
    light = pv.Light()

    exponent = 1.5
    light.exponent = exponent
    assert light.exponent == exponent

    cone_angle = 45
    light.cone_angle = cone_angle
    assert light.cone_angle == cone_angle

    attenuation_values = (3, 2, 1)
    light.attenuation_values = attenuation_values
    assert light.attenuation_values == attenuation_values


@pytest.mark.parametrize(
    ('int_code', 'enum_code'),
    [
        (1, pv.Light.HEADLIGHT),
        (2, pv.Light.CAMERA_LIGHT),
        (3, pv.Light.SCENE_LIGHT),
    ],
)
def test_type_properties(int_code, enum_code):
    light = pv.Light()

    # test that the int and enum codes match up
    assert int_code == enum_code

    # test that both codes work
    light.light_type = int_code
    assert light.light_type == int_code
    light.light_type = enum_code
    assert light.light_type == enum_code


def test_type_setters():
    light = pv.Light()

    light.set_headlight()
    assert light.is_headlight
    light.set_camera_light()
    assert light.is_camera_light
    light.set_scene_light()
    assert light.is_scene_light


def test_type_invalid():
    with pytest.raises(TypeError):
        light = pv.Light(light_type=['invalid'])
    with pytest.raises(ValueError):  # noqa: PT011
        light = pv.Light(light_type='invalid')

    light = pv.Light()

    with pytest.raises(TypeError):
        light.light_type = ['invalid']


def test_from_vtk():
    vtk_light = vtk.vtkLight()

    # set the vtk light
    for _, value, vtkname in configuration:
        vtk_setter = getattr(vtk_light, vtkname)
        # we can't pass the array to vtkLight directly
        input_value = pv.vtkmatrix_from_array(value) if isinstance(value, np.ndarray) else value
        vtk_setter(input_value)
    light = pv.Light.from_vtk(vtk_light)
    for pvname, value, _ in configuration:
        if isinstance(value, np.ndarray):
            trans_arr = pv.array_from_vtkmatrix(getattr(light, pvname))
            assert np.array_equal(trans_arr, value)
        else:
            assert getattr(light, pvname) == value

    # invalid case
    with pytest.raises(TypeError):
        pv.Light.from_vtk('invalid')
    with pytest.raises(TypeError):
        pv.Light(position='invalid')


def test_add_vtk_light():
    pl = pv.Plotter(lighting=None)
    pl.add_light(vtk.vtkLight())
    assert len(pl.renderer.lights) == 1


def test_actors():
    light = pv.Light()
    actor = light.actor

    # test showing
    assert not actor.GetVisibility()
    light.show_actor()
    assert not actor.GetVisibility()
    light.positional = True
    light.show_actor()
    assert actor.GetVisibility()

    # test hiding
    light.positional = False
    assert not actor.GetVisibility()
    light.positional = True
    light.show_actor()
    assert actor.GetVisibility()
    light.hide_actor()
    assert not actor.GetVisibility()
    light.show_actor()
    light.cone_angle = 90
    assert not actor.GetVisibility()
