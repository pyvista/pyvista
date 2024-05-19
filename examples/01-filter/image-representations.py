"""
.. _image_representations_example:

Image Data Representations
~~~~~~~~~~~~~~~~~~~~~~~~~~
This example demonstrates how to use :meth:`~pyvista.ImageDataFilters.points_to_cells`
and :meth:`~pyvista.ImageDataFilters.cells_to_points` to re-mesh :class:`~pyvista.ImageData`.

These filters can be used to ensure that image data has an appropriate representation
when generating plots and/or when using either point- or cell-based filters such as
:meth:`ImageDataFilters.image_threshold <pyvista.ImageDataFilters.image_threshold>` (point-based)
and :meth:`DataSetFilters.threshold <pyvista.DataSetFilters.threshold>` (cell-based).

"""

################################################################################
# Compare Representations
# -----------------------
# Create image data with eight points and a discrete scalar data array.

import numpy as np

import pyvista as pv

data_array = [8, 7, 6, 5, 4, 3, 1, 0]
points_image = pv.ImageData(dimensions=(2, 2, 2))
points_image.point_data['Data'] = data_array

################################################################################
# If we plot the image, it is represented as a single cell with eight points,
# and the point data is interpolated to color the cell.

points_image.plot(show_edges=True)

################################################################################
# However, in many applications (e.g. 3D medical images) the scalar data arrays
# represent discretized samples at the center-points of voxels. As such, it may
# be more appropriate to represent the data as eight cells instead of eight
# points. We can use :meth:`~pyvista.ImageDataFilters.points_to_cells` to
# generate a cell-based representation.

cells_image = points_image.points_to_cells()

################################################################################
# Now, when we plot the image, we have a more appropriate representation with
# eight voxel cells, and the scalar data is no longer interpolated.

cells_image.plot(show_edges=True)

################################################################################
# Let's compare the two representations and plot them together.
#
# For visualization, we color the points image (inner mesh) and show the cells
# image (outer mesh) as a wireframe. We also plot the cell centers in red. Note
# how the centers of the cells image correspond to the points of the points image.

cell_centers = cells_image.cell_centers()
cell_edges = cells_image.extract_all_edges()

plot = pv.Plotter()
plot.add_mesh(points_image, color=True, show_edges=True, opacity=0.7)
plot.add_mesh(cell_edges, color='black')
plot.add_points(
    cell_centers,
    render_points_as_spheres=True,
    color='red',
    point_size=20,
)
plot.camera.azimuth = -25
plot.camera.elevation = 25
plot.show()

################################################################################
# As long as only one kind of scalar data is used (i.e. either point or cell
# data, but not both), it is possible to move between representations without
# loss of data.

array_before = points_image.active_scalars
array_after = points_image.points_to_cells().cells_to_points().active_scalars
np.array_equal(array_before, array_after)

################################################################################
# Point Filters with Image Data
# -----------------------------
# Use a point representation of the image when working with point-based
# filters such as :meth:`~pyvista.ImageDataFilters.image_threshold`. If the
# image only has cell data, use :meth:`~pyvista.ImageDataFilters.cells_to_points`
# to re-mesh the input first.
#
# Here, we reuse the points image defined earlier and apply the filter. For
# context, we also show the input data array.

print(points_image.point_data['Data'])

dims = (4, 4, 4)
points_image = pv.ImageData(dimensions=dims)
points_image['Data'] = range(4*4*4)[::-1]
points_ithresh = points_image.image_threshold(2)

################################################################################
# This filter returns binary point data as expected. Values above the threshold
# of ``2`` are ones, and below the threshold are zeros.

print(points_ithresh.point_data['Data'])

################################################################################
# However, when we plot it the point values are interpolated as before.

points_ithresh.plot(show_edges=True)

################################################################################
# To better visualize the result, convert the points image returned by the
# filter to a cell representation with :meth:`~pyvista.ImageDataFilters.points_to_cells`
# before plotting.

points_ithresh_as_cells = points_ithresh.points_to_cells()
points_ithresh_as_cells.plot(show_edges=True)

################################################################################
# Cell Filters with Image Data
# ----------------------------
# Use a cell representation of the image when working with cell-based filters
# such as :meth:`~pyvista.DataSetFilters.threshold`. If the image only has point
# data, use :meth:`~pyvista.ImageDataFilters.points_to_cells` to re-mesh the
# input first.
#
# Here, we reuse the cells image created earlier and apply the filter. For
# context, we also show the input data array.

print(cells_image.cell_data['Data'])

cells_thresh = cells_image.threshold(2)

################################################################################
# When the input is cell data, this filter returns six discrete values above
# the threshold value of ``2`` as expected.

print(cells_thresh.cell_data['Data'])

cells_thresh.plot(show_edges=True)

################################################################################
# However, if we apply the same filter to a point-based representation of the
# image, the filter returns an unexpected result.

points_thresh = points_image.threshold(2)

################################################################################
# In this case, the filter has no effect on the data array's values.

print(points_thresh.point_data['Data'])

################################################################################
# If we plot the output, the result is identical to the plot of the input points
# image shown at the beginning of this example.

points_thresh.plot(show_edges=True)

################################################################################
# Representations of 2D Images
# ----------------------------
# The filters :meth:`~pyvista.ImageDataFilters.points_to_cells` and
# :meth:`~pyvista.ImageDataFilters.cells_to_points` can similarly be used
# with 2D images.
#
# For this example, we create a 4x4 2D grayscale image with 16 points to represent
# 16 pixels.

data_array = np.linspace(0, 255, 16, dtype=np.uint8)[::-1]
gray_points = pv.ImageData(dimensions=(4, 4, 1))
gray_points.point_data['Data'] = data_array

################################################################################
# Plot the image. As before, the plot does not appear correct since the point
# data is interpolated and nine cells are shown rather than the desired 16
# (one for each pixel).

plot_kwargs = dict(
    cpos='xy',
    zoom='tight',
    show_axes=False,
    cmap='gray',
    clim=[0, 255],
    show_edges=True,
)
gray_points.plot(**plot_kwargs)

################################################################################
# To visualize the image correctly, we first use :meth:`~pyvista.ImageDataFilters.points_to_cells`
# to get a cell-based representation of the image and plot the result. The plot
# now correctly shows 16 pixel cells with discrete values.

gray_cells = gray_points.points_to_cells()
gray_cells.plot(**plot_kwargs)
