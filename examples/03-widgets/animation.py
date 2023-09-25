"""
.. _animation_example:

Animation
~~~~~~~~~

This example demonstrates how to create a simple animation.
A timer is used to move a sphere across a scene.

Inspired by `VTK Animation Examples <https://examples.vtk.org/site/Python/Utilities/Animation/>`_.
"""

import pyvista as pv
import numpy as np

sphere = pv.Sphere()

pl = pv.Plotter()
actor = pl.add_mesh(sphere)


def callback(step):
    actor.position = [step / 100.0, step / 100.0, 0]


pl.add_timer_event(max_steps=200, duration=500, callback=callback)

cpos = [(0.0, 0.0, 10.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
pl.show(cpos=cpos)

###############################################################################
# Here is the example of plotting sin wave.

algo = pv.MultipleLinesSource(points=[[0.0, 0.0, 0.0], [0.0, np.sin(2.0 * np.pi / 200 * 1), 0.0]])

pl = pv.Plotter()
actor = pl.add_mesh(algo.output)

def callback(step):
    algo.points.append([0.0, np.sin(2.0 * np.pi / 200 * (step + 1)), 0.0])


pl.add_timer_event(max_steps=200, duration=500, callback=callback)

cpos = [(0.0, 0.0, 10.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
pl.show(cpos=cpos)


