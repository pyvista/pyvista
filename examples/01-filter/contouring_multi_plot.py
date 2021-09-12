"""
.. _contouring_along_axis_example:

Contouring Along Axis
~~~~~~~~~~~~~~~~~~~~~

Generate contour lines along an axis for the random hills dataset.

This uses :func:`slice_along_axis() <pyvista.DataSet.slice_along_axis>
to slice along the Z-Axis of the dataset.

"""

import numpy as np
import pyvista as pv
pv.set_plot_theme('document')
from pyvista import examples

###############################################################################
# Create the plotter and load the dataset
# 
pl = pv.Plotter(shape=(2, 2))

mesh_linear = examples.load_random_hills()
mesh_random = mesh_linear.copy()

###############################################################################
# Generate the data

# Linear Data
mesh_linear.point_data["DataLinear"] = np.linspace(
    0, 10, mesh_linear.points.shape[0]
)

# Random Data
rng = np.random.default_rng(seed=12345)
mesh_random.point_data["DataRandom"] = rng.integers(
    low=0, high=10, size=mesh_random.points.shape[0]
)

###############################################################################
# Add Plots
pl.subplot(0, 1)
mesh_linear.set_active_scalars("DataLinear")
pl.add_mesh(mesh_linear)

pl.subplot(1, 1)
mesh_random.set_active_scalars("DataRandom")
pl.add_mesh(mesh_random)

###############################################################################
# Create Averaged Data Arrays
mesh_linear.point_data["DataLinearContourAverages"] = np.zeros_like(
    mesh_linear["DataLinear"]
)
mesh_random.point_data["DataRandomContourAverages"] = np.zeros_like(
    mesh_random["DataRandom"]
)

# Slice Data Along Z-Azis
n_slices = 10

mesh_linear.set_active_scalars("Elevation")
mesh_random.set_active_scalars("Elevation")

contours_linear = mesh_linear.slice_along_axis(n=n_slices, axis="z")
contours_random = mesh_random.slice_along_axis(n=n_slices, axis="z")

###############################################################################
# Average Linear Data for Connected Component in Contour and Plot
for ndx, contour in enumerate(contours_linear):
    connectivity = contour.connectivity(largest=False)

    labels = connectivity.point_data["RegionId"]
    num_labels = len(np.unique(connectivity.point_data["RegionId"]))

    for id_ in range(num_labels):
        data = connectivity["DataLinear"][labels == id_]
        connectivity["DataLinearContourAverages"][labels == id_] = np.mean(data)

    pl.subplot(0, 0)
    connectivity.set_active_scalars("DataLinearContourAverages")
    pl.add_mesh(connectivity)

###############################################################################
# Average Random Data for Connected Component in Contour and Plot
for ndx, contour in enumerate(contours_random):
    connectivity = contour.connectivity(largest=False)

    labels = connectivity.point_data["RegionId"]
    num_labels = len(np.unique(connectivity.point_data["RegionId"]))

    for id_ in range(num_labels):
        data = connectivity["DataRandom"][labels == id_]
        connectivity["DataRandomContourAverages"][labels == id_] = np.mean(data)

    pl.subplot(1, 0)
    connectivity.set_active_scalars("DataRandomContourAverages")
    pl.add_mesh(connectivity)

mesh_linear.set_active_scalars("DataLinear")
mesh_random.set_active_scalars("DataRandom")

###############################################################################
# Show the plot

pl.show()
