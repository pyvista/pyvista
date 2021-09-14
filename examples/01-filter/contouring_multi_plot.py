"""
.. _contouring_along_axis_example:

Contouring Along Axis
~~~~~~~~~~~~~~~~~~~~~

Generate contour lines along an axis for the random hills dataset and
compute data averages within each connected contour line.

This uses :func:`slice_along_axis() <pyvista.DataSet.slice_along_axis>`
to slice along the Z-Axis of the dataset.

"""

import numpy as np
import pyvista as pv
from pyvista import examples


###############################################################################
# Load the dataset
mesh_linear = examples.load_random_hills()
mesh_random = mesh_linear.copy()

###############################################################################
# Generate the data

# linear data
mesh_linear.point_data["DataLinear"] = np.linspace(
    0, 10, mesh_linear.points.shape[0]
)

# random data
rng = np.random.default_rng(seed=12345)
mesh_random.point_data["DataRandom"] = rng.integers(
    low=0, high=10, size=mesh_random.points.shape[0]
)

###############################################################################
# Average data for each connected component.  For this we first generate
# contours as separate blocks in a :class:`MultiBlock <pyvista.MultiBlock>`,
# then for each contour dataset we find connected components and average the
# scalars within each connected component.  We do all these for both datasets.

# plot raw datasets side by side
pl = pv.Plotter(shape=(2, 2))

pl.subplot(0, 1)
mesh_linear.set_active_scalars("DataLinear")
pl.add_mesh(mesh_linear)

pl.subplot(1, 1)
mesh_random.set_active_scalars("DataRandom")
pl.add_mesh(mesh_random)

n_slices = 10

meshes = mesh_linear, mesh_random
data_names = "DataLinear", "DataRandom"

# plot averaged contours on the fly
for subplot_ind, (mesh, data_name) in enumerate(zip(meshes, data_names)):
    mesh.set_active_scalars("Elevation")
    contours = mesh.slice_along_axis(n=n_slices, axis="z")
    # contours is a MultiBlock with n_slices blocks

    for level_ind, contour in enumerate(contours):
        connectivity = contour.connectivity()
        # connectivity is annotated with "RegionId" that identifies components
        labels = connectivity.point_data["RegionId"]
        num_labels = np.unique(labels).size

        contour_data_name = data_name + "ContourAverages"
        connectivity.point_data[contour_data_name] = np.zeros_like(
            mesh.point_data[data_name], shape=labels.shape
        )

        for id_ in range(num_labels):
            data = connectivity[data_name][labels == id_]
            connectivity[contour_data_name][labels == id_] = np.mean(data)

        pl.subplot(subplot_ind, 0)
        connectivity.set_active_scalars(contour_data_name)
        pl.add_mesh(connectivity)

    mesh.set_active_scalars(data_name)

# show the plot
pl.link_views()
pl.view_isometric()
pl.show()
