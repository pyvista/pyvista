"""Module containing pyvista implementation of vtkLight."""

from enum import IntEnum

import vtk
from vtk import vtkLight

import pyvista
from .theme import parse_color
from ..utilities.helpers import vtkmatrix_from_array

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

    color : string or 3-length sequence, optional
        The color of the light. The ambient, diffuse and specular colors will
        all be set to this color on creation.

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

    Examples
    --------
    Create a light at (10, 10, 10) and set its diffuse color to red.

    >>> import pyvista as pv
    >>> light = pv.Light(position=(10, 10, 10))
    >>> light.diffuse_color = 1, 0, 0

    """

    # TODO: better/more explanation for ``position``?

    # pull in light type enum values as class constants
    HEADLIGHT = LightType.HEADLIGHT
    CAMERA_LIGHT = LightType.CAMERA_LIGHT
    SCENE_LIGHT = LightType.SCENE_LIGHT

    def __init__(self, position=None, color=None, light_type='scene light'):
        """Initialize the light."""
        super().__init__()

        if position:
            self.position = position

        if color:
            self.ambient_color = color
            self.diffuse_color = color
            self.specular_color = color

        if isinstance(light_type, str):
            # be forgiving: ignore spaces and case
            light_type_orig = light_type
            type_normalized = light_type.replace(' ', '').lower()
            mapping = {'headlight': LightType.HEADLIGHT,
                       'cameralight': LightType.CAMERA_LIGHT,
                       'scenelight': LightType.SCENE_LIGHT,
                      }
            try:
                light_type = mapping[type_normalized]
            except KeyError:
                raise ValueError(f'Invalid ``light_type`` "{light_type_orig}"') from None
        elif not isinstance(light_type, int):
            raise TypeError('Parameter ``light_type`` must be int or str,'
                            f' not {type(light_type)}.')
        # LightType is an int subclass

        self.light_type = light_type

        # TODO: should setting attenuation and cone angle automatically switch to positional?
        # TODO: should color and point and direction_angle have more flexible signatures? (only for non-properties)
        # TODO: ndarray type and shape and size checking for color and point
        # TODO: check if backticks in error messages are OK/necessary
        # TODO: examples, also for property getters!
        # TODO: add actor...?

    def __repr__(self):
        """Print a repr specifying the id of the light and its light type."""
        return (f'<{self.__class__.__name__} ({self.light_type}) at {hex(id(self))}>')

    def __eq__(self, other):
        """Compare whether the relevant attributes of two lights are equal."""
        # attributes which are native python types and thus implement __eq__
        native_attrs = [
            'light_type', 'position', 'focal_point', 'ambient_color',
            'diffuse_color', 'specular_color', 'intensity', 'on',
            'positional', 'exponent', 'cone_angle', 'attenuation_values',
            'shadow_attenuation',
        ]
        for attr in native_attrs:
            if getattr(self, attr) != getattr(other, attr):
                return False

        # check transformation matrix element by element (if it exists)
        this_trans = self.transform_matrix
        that_trans = other.transform_matrix
        trans_count = sum(1 for trans in [this_trans, that_trans] if trans is not None)
        if trans_count == 1:
            # either but not both are None
            return False
        if trans_count == 2:
            for i in range(4):
                for j in range(4):
                    if this_trans.GetElement(i, j) != that_trans.GetElement(i, j):
                        return False
        return True

    #### Properties ####

    @property
    def ambient_color(self):
        """Return the ambient color of the light.

        When setting, the color must be a 3-length sequence or a string.
        For example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1, 1, 1]``
            * ``color='#FFFFFF'``

        Examples
        --------
        Create a light and set its ambient color to red.

        >>> import pyvista as pv
        >>> light = pv.Light()
        >>> light.ambient_color = 'red'

        """
        return self.GetAmbientColor()

    @ambient_color.setter
    def ambient_color(self, color):
        """Set the ambient color of the light."""
        self.SetAmbientColor(parse_color(color))

    @property
    def diffuse_color(self):
        """Return the diffuse color of the light.

        When setting, the color must be a 3-length sequence or a string.
        For example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1, 1, 1]``
            * ``color='#FFFFFF'``

        Examples
        --------
        Create a light and set its diffuse color to blue.

        >>> import pyvista as pv
        >>> light = pv.Light()
        >>> light.diffuse_color = (0, 0, 1)

        """
        return self.GetDiffuseColor()

    @diffuse_color.setter
    def diffuse_color(self, color):
        """Set the diffuse color of the light."""
        self.SetDiffuseColor(parse_color(color))

    @property
    def specular_color(self):
        """Return the specular color of the light.

        When setting, the color must be a 3-length sequence or a string.
        For example:

            * ``color='white'``
            * ``color='w'``
            * ``color=[1, 1, 1]``
            * ``color='#FFFFFF'``

        Examples
        --------
        Create a light and set its specular color to bright green.

        >>> import pyvista as pv
        >>> light = pv.Light()
        >>> light.specular_color = '#00FF00'

        """
        return self.GetSpecularColor()

    @specular_color.setter
    def specular_color(self, color):
        """Set the specular color of the light."""
        self.SetSpecularColor(parse_color(color))

    @property
    def position(self):
        """Return the position of the light.

        Note: the position is defined in the coordinate space indicated
        by the light's transformation matrix (if it exists). To get the
        light's world space position, use the (read-only) :py:attr:`world_position`
        property.

        Examples
        --------
        Create a light positioned at (10, 10, 10) after initialization, and note
        how the position is unaffected by a non-trivial transform matrix.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> light = pv.Light()
        >>> light.position = (10, 10, 10)
        >>> # set a "random" transformation matrix
        >>> light.transform_matrix = np.arange(4 * 4).reshape(4, 4)
        >>> light.position
        (10.0, 10.0, 10.0)

        """
        return self.GetPosition()

    @position.setter
    def position(self, pos):
        """Set the position of the light."""
        self.SetPosition(pos)

    @property
    def world_position(self):
        """Return the world space position of the light.

        The world space position is the :py:attr:`position` property transformed by
        the light's transform matrix if it exists. The value of this read-only
        property corresponds to the ``vtk.vtkLight.GetTransformedPosition()`` method.

        Examples
        --------
        Create a light with a transformation matrix that corresponds to a
        90-degree rotation around the z axis and a shift by (0, 0, -1), and
        check that the light's position transforms as expected.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> light = pv.Light(position=(1, 0, 3))
        >>> trans = np.zeros((4, 4))
        >>> trans[:-1, :-1] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        >>> trans[:-1, -1] = [0, 0, -1]
        >>> light.transform_matrix = trans
        >>> light.position
        (1.0, 0.0, 3.0)
        >>> light.world_position
        (0.0, 1.0, 2.0)

        """
        return self.GetTransformedPosition()

    @property
    def focal_point(self):
        """Return the focal point of the light.

        Note: the focal point is defined in the coordinate space indicated
        by the light's transformation matrix (if it exists). To get the
        light's world space focal point, use the (read-only) :py:attr:`world_focal_point`
        property.

        Examples
        --------
        Create a light at (10, 10, 10) shining at (0, 0, 1).

        >>> import pyvista as pv
        >>> light = pv.Light(position=(10, 10, 10))
        >>> light.focal_point = (0, 0, 1)

        """
        return self.GetFocalPoint()

    @focal_point.setter
    def focal_point(self, pos):
        """Set the focal point of the light."""
        self.SetFocalPoint(pos)

    @property
    def world_focal_point(self):
        """Return the world space focal point of the light.

        The world space focal point is the :py:attr:`focal_point` property
        transformed by the light's transform matrix if it exists. The value of
        this read-only property corresponds to the
        ``vtk.vtkLight.GetTransformedFocalPoint()`` method.

        Examples
        --------
        Create a light with a transformation matrix that corresponds to a
        90-degree rotation around the z axis and a shift by (0, 0, -1), and
        check that the light's focal point transforms as expected.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> light = pv.Light()
        >>> light.focal_point = (1, 0, 3)
        >>> trans = np.zeros((4, 4))
        >>> trans[:-1, :-1] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        >>> trans[:-1, -1] = [0, 0, -1]
        >>> light.transform_matrix = trans
        >>> light.focal_point
        (1.0, 0.0, 3.0)
        >>> light.world_focal_point
        (0.0, 1.0, 2.0)

        """
        return self.GetTransformedFocalPoint()

    @property
    def intensity(self):
        """Return the brightness of the light (between 0 and 1)."""
        return self.GetIntensity()

    @intensity.setter
    def intensity(self, intensity):
        self.SetIntensity(intensity)

    @property
    def on(self):
        """Return whether the light is on.

        This corresponds to the Switch state of the ``vtk.vtkLight`` class.

        """
        return bool(self.GetSwitch())

    @on.setter
    def on(self, state):
        """Set whether the light should be on."""
        self.SetSwitch(state)

    @property
    def positional(self):
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

    @property
    def transform_matrix(self):
        """Return the transformation matrix of the light (if any).

        The transformation matrix is ``None`` by default, and it is stored
        as a ``vtk.vtkMatrix4x4`` object when set. If set, the light's
        parameters (position and focal point) are transformed by the matrix
        before being rendered. See also the :py:attr:`world_position` and
        :py:attr:`world_focal_point` read-only properties that can differ from
        :py:attr:`position` and :py:attr:`focal_point`, respectively.

        The 4-by-4 transformation matrix is a tool to encode a general linear
        transformation and a translation (an affine transform). The 3-by-3 principal
        submatrix (the top left corner of the matrix) encodes a three-dimensional
        linear transformation (e.g. some rotation around the origin). The top three
        elements in the last column of the matrix encode a three-dimensional
        translation. The last row of the matrix is redundant.

        Examples
        --------
        Create a light with a transformation matrix that corresponds to a
        90-degree rotation around the z axis and a shift by (0, 0, -1), and
        check that the light's position transforms as expected.

        >>> import numpy as np
        >>> import pyvista as pv
        >>> light = pv.Light(position=(1, 0, 3))
        >>> trans = np.zeros((4, 4))
        >>> trans[:-1, :-1] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        >>> trans[:-1, -1] = [0, 0, -1]
        >>> light.transform_matrix = trans
        >>> light.position
        (1.0, 0.0, 3.0)
        >>> light.world_position
        (0.0, 1.0, 2.0)

        """
        return self.GetTransformMatrix()

    @transform_matrix.setter
    def transform_matrix(self, matrix):
        """Set the 4x4 transformation matrix of the light."""
        if isinstance(matrix, vtk.vtkMatrix4x4):
            trans = matrix
        else:
            try:
                trans = vtkmatrix_from_array(matrix)
            except ValueError:
                raise ValueError('Transformation matrix must be '
                                 'a 4-by-4 matrix or array-like.') from None
        self.SetTransformMatrix(trans)

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
        # TODO: "Camera lights use the transform matrix to establish this space.": is this true in practice?
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

    def switch_on(self):
        """Switch on the light."""
        self.SwitchOn()

    def switch_off(self):
        """Switch off the light."""
        self.SwitchOff()

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

    def copy(self, deep=True):
        """Return a shallow or a deep copy of the light.

        The only mutable attribute of ``Light`` objects is the transformation
        matrix (if it exists). Thus asking for a shallow copy merely implies
        that the returned light and the original share the transformation
        matrix instance.

        Parameters
        ----------
        deep : bool
            Whether to return a deep copy rather than a shallow one. Default ``True``.

        """
        # let vtk do the heavy lifting
        if deep:
            other = vtkLight()
            other.DeepCopy(self)
        else:
            other = self.ShallowClone()
            # workaround until https://gitlab.kitware.com/vtk/vtk/-/issues/18093 is patched
            other.SetShadowAttenuation(self.shadow_attenuation)

        return Light.from_vtk(other)

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
        light.on = vtk_light.GetSwitch()
        light.positional = vtk_light.GetPositional()
        light.exponent = vtk_light.GetExponent()
        light.cone_angle = vtk_light.GetConeAngle()
        light.attenuation_values = vtk_light.GetAttenuationValues()
        light.shadow_attenuation = vtk_light.GetShadowAttenuation()
        trans = vtk_light.GetTransformMatrix()
        if trans is not None:
            light.transform_matrix = trans

        return light
