"""
.. _light_types_example:

Light Types
~~~~~~~~~~~

Lights come in three types:

  * headlights, the axis of which always coincides with the view of the camera,
  * camera lights, which move together with the camera, but which can occupy
    any fixed relative position with respect to the camera,
  * scene lights, the position of which is fixed to the scene, and which is thus
    unaffected by moving the camera. This is the default type.

Headlight
=========

For headlights the :py:attr:`pyvista.Camera.position` and
:py:attr:`pyvista.Camera.focal_point` properties are meaningless. No matter
where you move the camera, the light always emanates from the view point:

"""

# sphinx_gallery_thumbnail_number = 2
from __future__ import annotations

import pyvista as pv
from pyvista import examples

mesh = examples.download_bunny()
mesh.rotate_x(90, inplace=True)
mesh.rotate_z(180, inplace=True)

pl = pv.Plotter(lighting='none')
pl.add_mesh(mesh, color='lightblue', smooth_shading=True)
light = pv.Light(light_type='headlight')
# these don't do anything for a headlight:
light.position = (1, 2, 3)
light.focal_point = (4, 5, 6)
pl.add_light(light)
pl.show()


# %%
# Camera light
# ============
#
# Camera lights define their :py:attr:`pyvista.Camera.position` and
# :py:attr:`pyvista.Camera.focal_point` properties in a coordinate system that
# is local to the camera. The coordinates in the scene's coordinate system can
# be accessed through the :py:attr:`pyvista.Light.world_position` and
# :py:attr:`pyvista.Light.world_focal_point` read-only properties,
# respectively. For specifics of the local coordinate system used for the
# coordinates please see the documentation of
# :func:`pyvista.Light.set_camera_light`.

pl = pv.Plotter(lighting='none')
pl.add_mesh(mesh, color='lightblue', smooth_shading=True)
# a light that always shines from the right of the camera
light = pv.Light(position=(1, 0, 0), light_type='camera light')
pl.add_light(light)
pl.show()


# %%
# Scene light
# ===========
#
# Scene lights are attached to the scene, their position and focal point are
# interpreted as global coordinates:

pl = pv.Plotter(lighting='none')
pl.add_mesh(mesh, color='lightblue', smooth_shading=True)
# a light that always shines on the left side of the bunny
light = pv.Light(position=(0, 1, 0), light_type='scene light')
pl.add_light(light)
pl.show()
# %%
# .. tags:: lights
