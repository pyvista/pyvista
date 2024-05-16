"""

.. _image_representations_example:

Image Data Representations
~~~~~~~~~~~~~~~~~~~~~~~~~~

This example demonstrates how to use :meth:`~pyvista.ImageDataFilters.points_to_cells`.
and :meth:`~pyvista.ImageDataFilters.cells_to_points` to re-mesh :class:`~pyvista.ImageData`.

These filters are used to ensure that image data has the appropriate representation
when generating plots and/or when using either point- or cell-based filters such
as :meth:`~pyvista.ImageDataFilters.image_threshold` (point-based) and
:meth:`~pyvista.DataSetFilters.threshold` (cell-based).

"""

################################################################################
# Compare Representations
# ----------------------------------------
# Create image data with eight points and generate discrete scalar data.

import numpy as np

import pyvista as pv

points_image = pv.ImageData(dimensions=(2, 2, 2))
points_image.point_data['Data'] = list(range(8))[::-1]
points_image.n_points
points_image.n_cells
points_image.dimensions

################################################################################
# If we plot the image, it is represented as a single cell with eight points,
# and the point data is interpolated to color the cell.

points_image.plot(show_edges=True)

################################################################################
# However, in many applications (e.g. 3D medical images) the scalar data arrays
# represent discretized samples at the center-points of voxels. As such, it may
# be preferred to represent the data as eight cells instead of eight points. In
# this case, we can re-mesh the point data to cell data to create a cell-based
# representation.

cells_image = points_image.points_to_cells()
cells_image.n_points
cells_image.n_cells
cells_image.dimensions

################################################################################
# Now, when we plot the image, we have a more appropriate representation with
# eight voxel cells, and the scalar data is no longer interpolated.

cells_image.plot(show_edges=True)

################################################################################
# Show the two representations together. Note how the cell centers of the cells
# image correspond to the points of the points image.

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
# affecting the values of the scalar data.

points_image.active_scalars
points_image.points_to_cells().cells_to_points().active_scalars

################################################################################
# Using Point Filters with Image Data
# -----------------------------------
# With a point-based representation of the image, we can use a point-based
# filter such as :meth:`~pyvista.ImageDataFilters.image_threshold`.

points_thresh = points_image.image_threshold(2)

################################################################################
# The filter works as expected, but when we plot it the values are interpolated
# as before.

points_thresh.plot(show_edges=True)

################################################################################
# Convert the point-based output from the filter to a cell representation to
# better visualize the result.

cells_thresh = points_thresh.points_to_cells()
cells_thresh.plot(show_edges=True)

################################################################################
# Using Cell Filters with Image Data
# ----------------------------------
# With a cell-based representation of the image, we can use a cell-based filter
# such as :meth:`~pyvista.DataSetFilters.threshold`.

cells_thresh = cells_image.threshold(2)
cells_thresh.plot(show_edges=True)

################################################################################
# Using the cell representation with this filter produces the expected result
# since the original scalar data represents discrete voxels.
#
# For comparison, let's apply the same filter to the point-based representation.

points_thresh = points_image.threshold(2)
points_thresh.plot(show_edges=True)

################################################################################
# We can see that applying the filter to the point representation of the data
# produces a very different result than applying the same filter to the cell
# representation. In fact, the plot of the output (thresholded image) is
# identical to the plot of the input (points image) shown at the start of this
# example. Since the input points image only has a single cell, the cell-based
# filter had no effect on the data.

points_thresh['Data']
points_image['Data']

################################################################################
# Representations for 2D Images
# -----------------------------
# Create a 2D grayscale image with 16 points representing 16 pixels.

gray_points = pv.ImageData(dimensions=(4, 4, 1))
gray_points.point_data['Data'] = np.linspace(0, 255, 16, dtype=np.uint8)

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
# Plot the image as cells instead to show 16 pixels with discrete values.

gray_cells = gray_points.points_to_cells()
gray_cells.plot(**plot_kwargs)
