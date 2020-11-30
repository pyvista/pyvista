"""Module containing pyvista implementation of vtkLight."""

from enum import IntEnum

import vtk
from vtk import vtkLight

import pyvista

class LightType(IntEnum):
    """An enumeration for the light types."""

    HEADLIGHT = 1
    CAMERA_LIGHT = 2
    SCENE_LIGHT = 3

    def __str__(self):
        """Pretty name for a light type."""
        return self.name.replace('_', ' ').title()

class Light(vtkLight):
    """Light class.

    Parameters
    ----------
    position : list or tuple, optional
        The position of the light. The interpretation of the position depends
        on the type of the light.

    color : list or tuple, optional
        The color of the light. The ambient, diffuse and specular colors will
        all be set to this color in creation.

    light_type : string or int, optional
        The type of the light. If a string, one of ``'headlight'``,
        ``'camera light'`` or ``'scene light'``. If an int, one of 1, 2 or 3,
        respectively. The class constants ``Light.HEADLIGHT``, ``Light.CAMERA_LIGHT``
        and ``Light.SCENE_LIGHT`` are also available, respectively.

        A headlight is attached to the camera, looking at its focal point along
        the axis of the camera.
        A camera light also moves with the camera, but it can occupy a general
        position with respect to it.
        A scene light is stationary with respect to the scene, as it does not
        follow the camera. This is the default.
    """

    # TODO: better/more explanation for ``position``?

    # pull in light type enum values as class constants
    HEADLIGHT = LightType.HEADLIGHT
    CAMERA_LIGHT = LightType.CAMERA_LIGHT
    SCENE_LIGHT = LightType.SCENE_LIGHT
    # TODO: is this design OK?

    def __init__(self, position=None, color=None, light_type='scene light'):
        """Initialize the light."""
        super().__init__()

        if position:
            self.position = position

        if color:
            self.ambient_color = color
            self.diffuse_color = color
            self.specular_color = color
            # TODO: make this a single call if that's implemented

        if isinstance(light_type, str):
            # be forgiving: ignore spaces and case
            light_type_orig = light_type
            type_normalized = light_type.replace(' ', '').lower()
            mapping = {'headlight': LightType.HEADLIGHT,
                       'cameralight': LightType.CAMERA_LIGHT,
                       'scenelight': LightType.SCENE_LIGHT,
                      }
            light_type = mapping.get(type_normalized)
            if light_type is None:
                raise ValueError(f'Invalid ``light_type`` "{light_type_orig}"')
        elif not isinstance(light_type, int):
            raise TypeError('Parameter ``light_type`` must be int or str,'
                            f' not {type(light_type)}.')
        # LightType is an int subclass

        self.light_type = light_type

        # TODO: more init kwargs (focal_point, cone angle etc.)
        # TODO: default Positional state?
        # TODO: should setting attenuation and cone angle automatically switch to positional?
        # TODO: should color and point and direction_angle have more flexible signatures? (only for non-properties)
        # TODO: ndarray type and shape and size checking for color and point

    def __repr__(self):
        """Print a repr specifying the id of the light and its light type."""
        return (f'<{self.__class__.__name__} ({self.light_type}) at {hex(id(self))}>')

    #### Properties ####

    @property
    def ambient_color(self):
        """Return the ambient color of the light."""
        return self.GetAmbientColor()

    @ambient_color.setter
    def ambient_color(self, color):
        """Set the ambient color of the light."""
        self.SetAmbientColor(color)

    @property
    def diffuse_color(self):
        """Return the diffuse color of the light."""
        return self.GetDiffuseColor()

    @diffuse_color.setter
    def diffuse_color(self, color):
        """Set the diffuse color of the light."""
        self.SetDiffuseColor(color)

    @property
    def specular_color(self):
        """Return the specular color of the light."""
        return self.GetSpecularColor()

    @specular_color.setter
    def specular_color(self, color):
        """Set the specular color of the light."""
        self.SetSpecularColor(color)

    # TODO: implement light.color = ... using setattr? no guarantee that a getter would always make sense, so property won't work!

    @property
    def position(self):
        """Return the position of the light.

        Note: the position is defined in the coordinate space indicated
        by the light's transformation matrix (if it exists). To get the
        light's world space position, use the ``world_position`` property.

        """
        return self.GetPosition()

    @position.setter
    def position(self, pos):
        self.SetPosition(pos)

    @property
    def world_position(self):
        # TODO: is this name and configuration OK? Same for world_focal_point
        # TODO: can a transformation matrix happen accidentally? If not, perhaps we can just not expose these at all!
        """Return the world space position of the light."""
        return self.GetTransformedPosition()

    @property
    def focal_point(self):
        """Return the focal point of the light.

        Note: the focal point is defined in the coordinate space indicated
        by the light's transformation matrix (if it exists). To get the
        light's world space focal point, use the ``world_focal_point``
        property.

        """
        return self.GetFocalPoint()

    @focal_point.setter
    def focal_point(self, pos):
        self.SetFocalPoint(pos)

    @property
    def world_focal_point(self):
        """Return the world space focal point of the light."""
        return self.GetTransformedFocalPoint()

    @property
    def intensity(self):
        """Return the brightness of the light (between 0 and 1)."""
        return self.GetIntensity()

    @intensity.setter
    def intensity(self, intensity):
        self.SetIntensity(intensity)

    @property
    def is_on(self):
        # TODO: is this name OK? Just "on" sounds too short.
        """Return whether the light is on."""
        return bool(self.GetSwitch())

    @is_on.setter
    def is_on(self, state):
        """Set whether the light should be on."""
        self.SetSwitch(state)

    @property
    def positional(self):
        # TODO: did I understand this correctly?
        """Return whether the light is positional.

        The default is a directional light, i.e. an infinitely distant
        point source. Attenuation and cone angles are only used for a
        positional light.

        """
        return bool(self.GetPositional())

    @positional.setter
    def positional(self, state):
        """Set whether the light should be positional."""
        self.SetPositional(state)

    @property
    def exponent(self):
        """Return the exponent of the cosine used in positional lighting."""
        return self.GetExponent()

    @exponent.setter
    def exponent(self, exp):
        """Set the exponent of the cosine used in positional lighting."""
        self.SetExponent(exp)

    @property
    def cone_angle(self):
        """Return the cone angle of a positional light.

        The angle is in degrees and is measured between the axis of the cone
        and an extremal ray of the cone. A value smaller than 90 has spot
        lighting effects, anything equal to and above 90 is just a positional
        light.

        """
        return self.GetConeAngle()

    @cone_angle.setter
    def cone_angle(self, angle):
        """Set the cone angle of a positional light."""
        self.SetConeAngle(angle)

    @property
    def attenuation_values(self):
        """Return the quadratic attenuation constants.

        The values specify the constant, linear and quadratic constants
        in this order.

        """
        return self.GetAttenuationValues()

    @attenuation_values.setter
    def attenuation_values(self, values):
        """Set the quadratic attenuation constants."""
        self.SetAttenuationValues(values)

    # TODO: implement transformation_matrix here?

    @property
    def light_type(self):
        """Return the light type.

        The default light type is a scene light which lives in world
        coordinate space.

        A headlight is attached to the camera and always points at the
        camera's focal point.

        A camera light also moves with the camera, but it can have an
        arbitrary relative position to the camera. Camera lights are
        defined in a coordinate space where the camera is located at
        (0, 0, 1), looking towards (0, 0, 0) at a distance of 1, with
        up being (0, 1, 0). Camera lights use the transform matrix to
        establish this space.

        The property returns class constant values from an enum:
            - Light.HEADLIGHT == 1
            - Light.CAMERA_LIGHT == 2
            - Light.SCENE_LIGHT == 3

        """
        return LightType(self.GetLightType())

    @light_type.setter
    def light_type(self, ltype):
        """Set the light type.

        Either an integer code or a class constant enum value must be used.

        """
        if not isinstance(ltype, int):
            # note that LightType is an int subclass
            raise TypeError('Light type must be an integer subclass,'
                            f' got {ltype} instead.')
        self.SetLightType(ltype)

    @property
    def is_headlight(self):
        """Return whether the light is a headlight."""
        return bool(self.LightTypeIsHeadlight())

    @property
    def is_camera_light(self):
        """Return whether the light is a camera light."""
        return bool(self.LightTypeIsCameraLight())

    @property
    def is_scene_light(self):
        """Return whether the light is a scene light."""
        return bool(self.LightTypeIsSceneLight())

    @property
    def shadow_attenuation(self):
        """Return the shadow intensity.

        By default a light will be completely blocked when in shadow.
        By setting this value to less than 1 you can control how much
        light is attenuated when in shadow.

        """
        return self.GetShadowAttenuation()

    @shadow_attenuation.setter
    def shadow_attenuation(self, shadow_intensity):
        """Set the shadow intensity."""
        self.SetShadowAttenuation(shadow_intensity)

    #### Everything else ####

    def set_color(self, color):
        # TODO: do we want to remove this?
        # TODO: do we want instead to set _all_ lights in pyvista?
        # TODO: do we want to support a three-arg form?
        """Set the diffuse and specular color of the light.

        Parameters
        ----------
        color : tuple(float)
            The color that should be set for diffuse and specular.

        """
        self.SetColor(color)

    def switch_on(self):
        """Switch on the light."""
        self.SwitchOn()

    def switch_off(self):
        """Switch off the light."""
        self.SwitchOff()

    def switch(self, is_on=None):
        """Set whether the light should be on. Toggles by default.

        Parameters
        ----------
        is_on : bool
            The state of the light to be set (True is on, False is off).
            Default is to toggle the state.

        """
        if is_on is None:
            # toggle
            is_on = not self.is_on
        self.SetSwitch(is_on)

    def positional_on(self):
        """Make the light positional.

        Attenuation and cone angles are only used for a positional light.

        """
        self.PositionalOn()

    def positional_off(self):
        """Make the light directional.

        Attenuation and cone angles are ignored for a directional light.

        """
        self.PositionalOff()

    # TODO: implement transform_point, transform_vector here?

    def set_direction_angle(self, elev, azim):
        """Set the position and focal point of a directional light.

        The light is switched into directional (non-positional). The
        position and focal point can be defined in terms of an elevation
        and an azimuthal angle, both in degrees.

        Parameters
        ----------
        elev : float
            The elevation of the directional light.

        azim : float
            The azimuthal angle of the directional light.

        """
        self.SetDirectionAngle(elev, azim)

    # TODO: deepcopy?

    def set_headlight(self):
        """Set the light to be a headlight.

        Headlights are fixed to the camera and always point to the focal
        point of the camera. Calling this method will reset the light's
        transformation matrix.

        """
        self.SetLightTypeToHeadlight()

    def set_camera_light(self):
        """Set the light to be a camera light.

        Camera lights are fixed to the camera and always point to the focal
        point of the camera.

        A camera light moves with the camera, but it can have an arbitrary
        relative position to the camera. Camera lights are defined in a
        coordinate space where the camera is located at (0, 0, 1), looking
        towards (0, 0, 0) at a distance of 1, with up being (0, 1, 0).
        Camera lights use the transform matrix to establish this space.
        Calling this method will reset the light's transformation matrix.

        """
        self.SetLightTypeToCameraLight()

    def set_scene_light(self):
        """Set the light to be a scene light.

        Scene lights are stationary with respect to the scene.
        Calling this method will reset the light's transformation matrix.

        """
        self.SetLightTypeToSceneLight()

    @classmethod
    def from_vtk(cls, vtk_light):
        """Create a Light object from a vtkLight, resulting in a copy.

        Parameters
        ----------
        vtk_light : vtkLight
            The ``vtkLight`` to be copied.

        """
        if not isinstance(vtk_light, vtkLight):
            raise TypeError(f'Expected ``vtkLight`` object, got ``{type(vtk_light)}`` instead.')

        light = cls()
        light.light_type = vtk_light.GetLightType()  # resets transformation matrix!
        light.position = vtk_light.GetPosition()
        light.focal_point = vtk_light.GetFocalPoint()
        light.ambient_color = vtk_light.GetAmbientColor()
        light.diffuse_color = vtk_light.GetDiffuseColor()
        light.specular_color = vtk_light.GetSpecularColor()
        light.intensity = vtk_light.GetIntensity()
        light.is_on = vtk_light.GetSwitch()
        light.positional = vtk_light.GetPositional()
        light.exponent = vtk_light.GetExponent()
        light.cone_angle = vtk_light.GetConeAngle()
        light.attenuation_values = vtk_light.GetAttenuationValues()
        light.shadow_attenuation = vtk_light.GetShadowAttenuation()
        # TODO: copy transformation matrix even if not exposed?

        return light
