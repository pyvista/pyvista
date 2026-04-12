"""
.. _attenuation_example:

Attenuation
~~~~~~~~~~~

This example shows how use :attr:`~pyvista.Light.attenuation_values`.

Attenuation is the phenomenon of light's intensity being gradually dampened as
it propagates through a medium. In PyVista positional lights can show attenuation.
The quadratic attenuation model uses three parameters to describe attenuation:
a constant, a linear and a quadratic parameter. These parameters
describe the decrease of the beam intensity as a function of the distance, `I(r)`.
In a broad sense the constant, linear and quadratic components correspond to
`I(r) = 1`, `I(r) = 1/r` and `I(r) = 1/r^2` decay of the intensity with distance
from the point source. In all cases a larger attenuation value (of a given kind)
means stronger dampening (weaker light at a given distance).

So the constant attenuation parameter corresponds roughly to a constant intensity
component. The linear and the quadratic attenuation parameters correspond to intensity
components that decay with distance from the source. For the same parameter value the
quadratic attenuation produces a beam that is shorter in range than that produced
by linear attenuation.

Three spotlights with three different attenuation profiles each:
"""

# sphinx_gallery_thumbnail_number = 3
from __future__ import annotations

import pyvista as pv

pl = pv.Plotter(lighting='none')
billboard = pv.Plane(direction=(1, 0, 0), i_size=6, j_size=6)
pl.add_mesh(billboard, color='white')

all_attenuation_values = [(1, 0, 0), (0, 2, 0), (0, 0, 2)]
offsets = [-2, 0, 2]
for attenuation_values, offset in zip(all_attenuation_values, offsets, strict=True):
    light = pv.Light(
        position=(0.1, offset, 2), focal_point=(0.1, offset, 1), color='cyan'
    )
    light.positional = True
    light.cone_angle = 20
    light.intensity = 15
    light.attenuation_values = attenuation_values
    pl.add_light(light)

pl.view_yz()
pl.show()


# %%
# It's not too obvious but it's visible that the rightmost light with quadratic
# attenuation has a shorter range than the middle one with linear attenuation.
# Although it seems that even the leftmost light with constant attenuation loses
# its brightness gradually, this partly has to do with the fact that we sliced
# the light beams very close to their respective axes, meaning that light hits
# the surface in a very small angle. Altering the scene such that the lights
# are further away from the plane changes this:

pl = pv.Plotter(lighting='none')
billboard = pv.Plane(direction=(1, 0, 0), i_size=6, j_size=6)
pl.add_mesh(billboard, color='white')

all_attenuation_values = [(1, 0, 0), (0, 2, 0), (0, 0, 2)]
offsets = [-2, 0, 2]
for attenuation_values, offset in zip(all_attenuation_values, offsets, strict=True):
    light = pv.Light(
        position=(0.5, offset, 3), focal_point=(0.5, offset, 1), color='cyan'
    )
    light.positional = True
    light.cone_angle = 20
    light.intensity = 15
    light.attenuation_values = attenuation_values
    pl.add_light(light)

pl.view_yz()
pl.show()

# %%
# Now the relationship of the three kinds of attenuation seems clearer.
#
# For a more practical comparison, let's look at planes that are perpendicular
# to the axis of each light (making use of the fact that shadowing between
# objects is not handled by default):

pl = pv.Plotter(lighting='none')

# loop over three lights with three kinds of attenuation
all_attenuation_values = [(2, 0, 0), (0, 2, 0), (0, 0, 2)]
light_offsets = [-6, 0, 6]
for attenuation_values, light_x in zip(
    all_attenuation_values, light_offsets, strict=True
):
    # loop over three perpendicular planes for each light
    for plane_y in [2, 5, 10]:
        screen = pv.Plane(
            center=(light_x, plane_y, 0), direction=(0, 1, 0), i_size=5, j_size=5
        )
        pl.add_mesh(screen, color='white')

    light = pv.Light(position=(light_x, 0, 0), focal_point=(light_x, 1, 0), color='cyan')
    light.positional = True
    light.cone_angle = 15
    light.intensity = 5
    light.attenuation_values = attenuation_values
    light.show_actor()
    pl.add_light(light)

pl.view_vector((1, -2, 2))
pl.show()
# %%
# .. tags:: lights
