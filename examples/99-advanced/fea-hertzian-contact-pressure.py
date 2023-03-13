import numpy as np
import matplotlib.pyplot as plt

import os
import pyvista as pv
from pyvista import examples


###############################################################################
# Load the dataset and plot it by "PartID"


###############################################################################
# Plot the DataSet
# ~~~~~~~~~~~~~~~~

mesh.plot(scalars='PartID', cmap=['green', 'blue'], show_scalar_bar=False)


###############################################################################
# Make two points to construct a line denoting the point of contact between the
# cylinder and the sphere
ypos = 0.024
a = [0.1, ypos, 0.2 - 1E-4]
b = [0.095, ypos, 0.2 - 1E-4]
line = pv.Line(a, b, resolution=100)
line.clear_data()


###############################################################################
# Plot the line and the dataset

# pl = pv.Plotter()
# pl.add_mesh(line, style='wireframe', color='red', line_width=10)
# pl.add_mesh(mesh, scalars='PartID', cmap=['green', 'blue'])
# pl.camera_position = 'iso'
# pl.set_focus(a)
# pl.camera.zoom(4)
# pl.show()


###############################################################################
# Sample along the Line
# ~~~~~~~~~~~~~~~~~~~~~
# Sample along the contact edge and compare with expected pressures.

sampled = line.sample(mesh, tolerance=1E-3)
x_coord = 0.1 - sampled.points[:, 0]
samp_stress = -sampled['Stress'][:, 2]

plt.plot(x_coord, samp_stress, '.')

expected = np.loadtxt('Hertzian_contact_pressure_F201152.csv', delimiter=',')


plt.plot(expected[:, 0], expected[:, 1])
plt.show()


###############################################################################
pl = pv.Plotter()
pl.add_mesh(sampled, style='wireframe', line_width=10, clim=[-1.8E9, -1.5E9], cmap='jet_r', scalars='Stress', component=2)
z_stress = np.abs(mesh['Stress'][:, 2])
pl.add_mesh(mesh,
            # scalars='vonMises',
            scalars=z_stress,
            clim=[0, 1.2E9],
            # cmap='bwr',
            cmap='jet',
            # color='w',
            lighting=True,
            show_edges=False,
            # split_sharp_edges=True,
            # smooth_shading=True,
            ambient=0.2,
)
pl.camera_position = 'xz'
# pl.set_focus(a)
# pl.camera.zoom(20)
pl.show()
