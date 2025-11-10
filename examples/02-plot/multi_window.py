"""
.. _multi_window_example:

Multi-Window Plot
~~~~~~~~~~~~~~~~~


Subplotting: having multiple scenes in a single window
"""

from __future__ import annotations

import pyvista as pv
from pyvista import examples

# sphinx_gallery_start_ignore
# labels are not supported in vtk-js
PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = True
# sphinx_gallery_end_ignore


# %%
# This example shows how to create a multi-window plotter by specifying the
# ``shape`` parameter.  The window generated is a two by two window by setting
# ``shape=(2, 2)``. Use the :func:`pyvista.Plotter.subplot` method to
# select the subplot you wish to be the active subplot.

pl = pv.Plotter(shape=(2, 2))

pl.subplot(0, 0)
pl.add_text('Render Window 0', font_size=30)
globe = examples.load_globe()
texture = examples.load_globe_texture()
pl.add_mesh(globe, texture=texture)

pl.subplot(0, 1)
pl.add_text('Render Window 1', font_size=30)
pl.add_mesh(pv.Cube(), show_edges=True, color='lightblue')

pl.subplot(1, 0)
pl.add_text('Render Window 2', font_size=30)
sphere = pv.Sphere()
pl.add_mesh(sphere, scalars=sphere.points[:, 2])
pl.add_scalar_bar('Z')
# pl.add_axes()
pl.add_axes(interactive=True)

pl.subplot(1, 1)
pl.add_text('Render Window 3', font_size=30)
pl.add_mesh(pv.Cone(), color='g', show_edges=True)
pl.show_bounds(all_edges=True)

# Display the window
pl.show()


# %%
pl = pv.Plotter(shape=(1, 2))

# Note that the (0, 0) location is active by default
# load and plot an airplane on the left half of the screen
pl.add_text('Airplane Example\n', font_size=30)
pl.add_mesh(examples.load_airplane(), show_edges=False)

# load and plot the uniform data example on the right-hand side
pl.subplot(0, 1)
pl.add_text('Uniform Data Example\n', font_size=30)
pl.add_mesh(examples.load_uniform(), show_edges=True)

# Display the window
pl.show()


# %%
# Split the rendering window in half and subdivide it in a nr. of vertical or
# horizontal subplots.

# This defines the position of the vertical/horizontal splitting, in this
# case 40% of the vertical/horizontal dimension of the window
pv.global_theme.multi_rendering_splitting_position = 0.40

# shape="3|1" means 3 plots on the left and 1 on the right,
# shape="4/2" means 4 plots on top of 2 at bottom.
pl = pv.Plotter(shape='3|1', window_size=(1000, 1200))

pl.subplot(0)
pl.add_text('Airplane Example')
pl.add_mesh(examples.load_airplane(), show_edges=False)

# load and plot the uniform data example on the right-hand side
pl.subplot(1)
pl.add_text('Uniform Data Example')
pl.add_mesh(examples.load_uniform(), show_edges=True)

pl.subplot(2)
pl.add_text('A Sphere')
pl.add_mesh(pv.Sphere(), show_edges=True)

pl.subplot(3)
pl.add_text('A Cone')
pl.add_mesh(pv.Cone(), show_edges=True)

# Display the window
pl.show()


# %%
# To get full flexibility over the layout grid, you can define the relative
# weighting of rows and columns and register groups that can span over multiple
# rows and columns. A group is defined through a tuple ``(rows,cols)`` of row
# and column indices or slices. The group always spans from the smallest to the
# largest (row or column) id that is passed through the list or slice.

# numpy is imported for a more convenient slice notation through np.s_
import numpy as np

shape = (5, 4)  # 5 by 4 grid
# First row is half the size and fourth row is double the size of the other rows
row_weights = [0.5, 1, 1, 2, 1]
# Third column is half the size and fourth column is double size of the other columns
col_weights = [1, 1, 0.5, 2]
groups = [
    (0, np.s_[:]),  # First group spans over all columns of the first row (0)
    ([1, 3], 0),  # Second group spans over row 1-3 of the first column (0)
    (np.s_[2:], [1, 2]),  # Third group spans over rows 2-4 and columns 1-2
    (slice(1, -1), 3),  # Fourth group spans over rows 1-3 of the last column (3)
]

pl = pv.Plotter(shape=shape, row_weights=row_weights, col_weights=col_weights, groups=groups)

# A grouped subplot can be activated through any of its composing cells using
# the subplot() method.

# Access all subplots and groups and plot something:
pl.subplot(0, 0)
pl.add_text('Group 1')
pl.add_mesh(pv.Cylinder(direction=[0, 1, 0], height=20))
pl.view_yz()
pl.camera.zoom(10)

pl.subplot(2, 0)
pl.add_text('Group 2')
pl.add_mesh(pv.ParametricCatalanMinimal(), show_edges=False, color='lightblue')
pl.view_isometric()
pl.camera.zoom(2)

pl.subplot(2, 1)
pl.add_text('Group 3')
pl.add_mesh(examples.load_uniform(), show_edges=True)

pl.subplot(1, 3)
pl.add_text('Group 4')
globe = examples.load_globe()
texture = examples.load_globe_texture()
pl.add_mesh(globe, texture=texture)

pl.subplot(1, 1)
pl.add_text('Cell (1,1)')
sphere = pv.Sphere()
pl.add_mesh(sphere, scalars=sphere.points[:, 2])
pl.add_scalar_bar('Z')
pl.add_axes(interactive=True)

pl.subplot(1, 2)
pl.add_text('Cell (1,2)')
pl.add_mesh(pv.Cone(), show_edges=True)

pl.subplot(4, 0)
pl.add_text('Cell (4,0)')
pl.add_mesh(examples.load_airplane(), show_edges=False)

pl.subplot(4, 3)
pl.add_text('Cell (4,3)')
pl.add_mesh(pv.Cube(), show_edges=True, color='lightblue')

# Display the window
pl.show()
# %%
# .. tags:: plot
