"""
.. _animation_example:

Animation
~~~~~~~~~

This example demonstrates how to create a simple animation.
A timer is used to move a sphere across a scene.

Inspired by https://examples.vtk.org/site/Python/Utilities/Animation/.
"""

import pyvista as pv


class TimerCallback:
    def __init__(self, steps, actor, iren):
        self.timer_count = 0
        self.steps = steps
        self.actor = actor
        self.iren = iren
        self.timer_id = None

    def execute(self, obj, event):
        step = 0
        while step < self.steps:
            self.actor.SetPosition(self.timer_count / 100.0, self.timer_count / 100.0, 0)
            iren = obj
            iren.GetRenderWindow().Render()
            self.timer_count += 1
            step += 1
        if self.timer_id:
            iren.DestroyTimer(self.timer_id)


sphere = pv.Sphere()

pl = pv.Plotter()
actor = pl.add_mesh(sphere)

cb = TimerCallback(200, actor, pl.iren)
obs_enter = pl.iren.add_observer("TimerEvent", cb.execute)
cb.timer_id = pl.iren.create_timer(500)

cpos = [(0.0, 0.0, 10.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
pl.show(cpos=cpos)
