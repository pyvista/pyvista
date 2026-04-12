"""
.. _fea_hertzian_contact_pressure_example:

Visualize Hertzian Contact Stress
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The following example demonstrates how to use PyVista to visualize
Hertzian contact stress between a cylinder and a flat plate.

This example loads a dataset, constructs a line to represent the point of
contact between the cylinder and the block, and samples the stress along that
line. Finally, it plots the dataset and the stress distribution.

**Background**
Hertzian contact stress refers to the stress that occurs between two curved
surfaces that are in contact with each other. It is named after Heinrich Rudolf
Hertz, a German physicist who first described the phenomenon in the late
1800s. Hertzian contact stress is an important concept in materials science,
engineering, and other fields where the behavior of materials under stress is a
critical consideration.

"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from pyvista import examples

# %%
# Load the dataset
# ~~~~~~~~~~~~~~~~
# Start by loading the dataset from the examples module with
# :func:`~pyvista.examples.downloads.download_fea_hertzian_contact_cylinder`.
# This module provides access to a range of datasets, including FEA
# (finite element analysis) datasets that are useful for stress analysis.

mesh = examples.download_fea_hertzian_contact_cylinder()
mesh


# %%
# Plot the Dataset
# ~~~~~~~~~~~~~~~~
# Plot the dataset by part ID.

mesh.plot(scalars='PartID', cmap=['green', 'blue'], show_scalar_bar=False)


# %%
# Creating a Line to Denote the Point of Contact
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Construct a line to represent the point of contact between the cylinder and
# the plate.

ypos = 0.024
a = [0.1, ypos, 0.2 - 1e-4]
b = [0.095, ypos, 0.2 - 1e-4]
line = pv.Line(a, b, resolution=100)
line.clear_data()
line

# %%
# Sampling the Stress along the Line
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We can sample the Z component stress along the contact edge and compare it
# with expected pressure.
#
# The expected values array is the Hertzian contact pressure and is the
# analytical solution to the non-adhesive contact problem. Computation of these
# values is an exercise left up to the reader (the radius of the cylinder is
# 0.05). See `Contact Mechanics
# <https://en.wikipedia.org/wiki/Contact_mechanics>`_

# Sample the stress
sampled = line.sample(mesh, tolerance=1e-3)
x_coord = 0.1 - sampled.points[:, 0]
samp_z_stress = -sampled['Stress'][:, 2]

# Expected Hertzian contact pressure
h_pressure = np.array(
    [
        [0.0000, 1718094092],
        [0.0002, 1715185734],
        [0.0004, 1703502649],
        [0.0006, 1683850714],
        [0.0008, 1655946243],
        [0.001, 1619362676],
        [0.0012, 1573494764],
        [0.0014, 1517500856],
        [0.0016, 1450208504],
        [0.0018, 1369953775],
        [0.002, 1274289906],
        [0.0022, 1159408887],
        [0.0024, 1018830677],
        [0.0026, 839747409.8],
        [0.0028, 587969605.2],
        [0.003, 0],
        [0.005, 0],
    ],
)

plt.plot(x_coord, samp_z_stress, '.', label='Z Component Stress')
plt.plot(h_pressure[:, 0], h_pressure[:, 1], label='Hertzian contact pressure')
plt.legend()
plt.show()


# %%
# Visualizing the Z Stress Distribution
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# You can now visualize the Z stress distribution. Use :class:`pyvista.Plotter` to
# create a plot window and add the dataset to it.

pl = pv.Plotter()
z_stress = np.abs(mesh['Stress'][:, 2])
pl.add_mesh(
    mesh,
    scalars=z_stress,
    clim=[0, 1.2e9],
    cmap='gouldian',
    scalar_bar_args={'title': 'Z Component Stress (Pa)', 'color': 'w'},
    lighting=True,
    show_edges=False,
    ambient=0.2,
)
pl.camera_position = 'xz'
pl.set_focus(a)
pl.camera.zoom(2.5)
pl.show()
