"""
.. _spline_widget_example:

Spline Widget
~~~~~~~~~~~~~


A spline widget can be enabled and disabled by the
:func:`pyvista.Plotter.add_spline_widget` and
:func:`pyvista.Plotter.clear_spline_widgets` methods respectively.
This widget allows users to interactively create a poly line (spline) through
a scene and use that spline.

A common task with splines is to slice a volumetric dataset using an irregular
path. To do this, we have added a convenient helper method which leverages the
:func:`pyvista.DataObjectFilters.slice_along_line` filter named
:func:`pyvista.Plotter.add_mesh_slice_spline`.
"""

from __future__ import annotations

import numpy as np

import pyvista as pv

# %%

mesh = pv.Wavelet()

# initial spline to seed the example
points = np.array(
    [
        [-8.64208925, -7.34294559, -9.13803458],
        [-8.25601497, -2.54814702, 0.93860914],
        [-0.30179377, -3.21555997, -4.19999019],
        [3.24099167, 2.05814768, 3.39041509],
        [4.39935227, 4.18804542, 8.96391132],
    ],
)

pl = pv.Plotter()
pl.add_mesh(mesh.outline(), color='black')
pl.add_mesh_slice_spline(mesh, initial_points=points, n_handles=5)
pl.camera_position = pv.CameraPosition(
    position=(30, -42, 30), focal_point=(0.0, 0.0, 0.0), viewup=(-0.09, 0.53, 0.84)
)
pl.show()
# %%
# .. tags:: widgets
