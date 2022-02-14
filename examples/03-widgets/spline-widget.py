"""
Spline Widget
~~~~~~~~~~~~~


A spline widget can be enabled and disabled by the
:func:`pyvista.WidgetHelper.add_spline_widget` and
:func:`pyvista.WidgetHelper.clear_spline_widgets` methods respectively.
This widget allows users to interactively create a poly line (spline) through
a scene and use that spline.

A common task with splines is to slice a volumetric dataset using an irregular
path. To do this, we have added a convenient helper method which leverages the
:func:`pyvista.DataSetFilters.slice_along_line` filter named
:func:`pyvista.WidgetHelper.add_mesh_slice_spline`.
"""
import numpy as np

import pyvista as pv

##############################################################################

mesh = pv.Wavelet()

# initial spline to seed the example
points = np.array(
    [
        [-8.64208925, -7.34294559, -9.13803458],
        [-8.25601497, -2.54814702, 0.93860914],
        [-0.30179377, -3.21555997, -4.19999019],
        [3.24099167, 2.05814768, 3.39041509],
        [4.39935227, 4.18804542, 8.96391132],
    ]
)

p = pv.Plotter()
p.add_mesh(mesh.outline(), color='black')
p.add_mesh_slice_spline(mesh, initial_points=points, n_handles=5)
p.camera_position = [(30, -42, 30), (0.0, 0.0, 0.0), (-0.09, 0.53, 0.84)]
p.show()
