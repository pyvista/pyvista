Lights
======
The :class:`pyvista.Light` class adds additional functionality and a pythonic API
to the ``vtk.vtkLight`` class. :class:`pyvista.Plotter` objects come with a default
set of lights that work well in most cases, but in many situations a more hands-on
access to lighting is necessary.


Brief Example
-------------

Create a red spotlight that shines on the origin, then create a scene
without lighting and add our light to it manually.

.. jupyter-execute::

    import pyvista as pv
    from pyvista import examples
    light = pv.Light(position=(-1, 1, 1), color='red')
    light.positional = True

    import pyvista as pv
    from pyvista import examples
    plotter = pv.Plotter(lighting='none')
    plotter.background_color = 'white'
    mesh = examples.download_bunny()
    mesh.rotate_x(90)
    mesh.rotate_z(180)
    plotter.add_mesh(mesh, specular=1.0, diffuse=0.7, smooth_shading=True)
    plotter.add_light(light)
    plotter.show()

For detailed examples please see :ref:`ref_light_examples`.


Light API
---------
``pyvista.Light`` instances come in three types: headlights, camera lights, and
scene lights. Headlights always shine along the camera's axis, camera lights
have a fixed position with respect to the camera, and scene lights are positioned
with respect to the scene, such that moving around the camera doesn't affect the
lighting of the scene.

Lights have a :py:attr:`position` and a :py:attr:`focal_point` that define the
axis of the light. The meaning of these depends on the type of the light. The
color of the light can be set according to ambient, diffuse and specular components.
The brightness can be set with the :py:attr:`intensity` property, and the writable
:py:attr:`on` property specifies whether the light is switched on.

Lights can be either directional (meaning an infinitely distant point source) or
:py:attr:`positional`. Positional lights have additional properties that describe
the geometry and the spatial distribution of the light. The :py:attr:`cone_angle`
and :py:attr:`exponent` properties define the shape of the light beam and the
angular distribution of the light's intensity within that beam. The fading of the
light with distance can be customized with the :py:attr:`attenuation_values` property.
Positional lights can also make use of an actor that represents the shape and color
of the light using a wireframe, see :func:`show_actor`.

Positional lights with a :py:attr:`cone_angle` of less than 90 degrees are known as
spotlights. Spotlights are unidirectional and they make full use of beam shaping
properties, namely :py:attr:`exponent` and attenuation.  Non-spotlight positional
lights, however, act like point sources located in the real-world position of the
light, shining in all directions of space. They display attenuation with distance
from the source, but their beam is isotropic in space. In contrast, directional
lights act as infinitely distant point sources, so they are unidirectional but they do
not attenuate.

API reference
~~~~~~~~~~~~~

.. autoclass:: pyvista.Light
   :members:
