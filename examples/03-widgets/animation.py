"""
.. _animation_example:

Animation
~~~~~~~~~

This example demonstrates how to create a simple animation.
A timer is used with :meth:`~pyvista.Plotter.add_timer_event`
to move a sphere across a scene.

Inspired by `VTK Animation Examples <https://examples.vtk.org/site/Python/Utilities/Animation/>`_.
"""

from __future__ import annotations

import pyvista as pv

sphere = pv.Sphere()

pl = pv.Plotter()
actor = pl.add_mesh(sphere)


def callback(step):
    actor.position = [step / 100.0, step / 100.0, 0]


pl.add_timer_event(max_steps=200, duration=500, callback=callback)

cpos = pv.CameraPosition(
    position=(0.0, 0.0, 10.0), focal_point=(0.0, 0.0, 0.0), viewup=(0.0, 1.0, 0.0)
)
pl.show(cpos=cpos)
# %%
# .. tags:: widgets
