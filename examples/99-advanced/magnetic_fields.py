"""
.. _magnetic_fields_example:

Plot a Magnetic Field
---------------------

The following example demonstrates how PyVista can be used to plot a magnetic
field.

This example relies on :func:`~pyvista.DataSetFilters.streamlines_from_source` to
generate streamlines and :func:`add_volume() <pyvista.Plotter.add_volume>` to plot
the strength of the magnetic field.

This dataset was created from the `Coil Field Lines
<https://magpylib.readthedocs.io/en/stable/_pages/user_guide/examples/examples_app_coils.html>`_
example from the awesome `magpylib <https://github.com/magpylib/magpylib>`_
library.

"""

# sphinx_gallery_thumbnail_number = 3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "numpy",
#   "pyvista",
# ]
# ///

from __future__ import annotations

import numpy as np

import pyvista as pv
from pyvista import examples

# %%
# Download the DataSet
# ~~~~~~~~~~~~~~~~~~~~
# Let's first download the example dataset and show that it's a
# :class:`pyvista.ImageData` with the magnetic field stored as the ``'B'``
# array in ``point_data``.

grid = examples.download_coil_magnetic_field()
grid.point_data


# %%
# Create Coils
# ~~~~~~~~~~~~
# Create several hoops to represent the coil. This matches the geometry in the
# original example.

coils = [
    pv.Polygon(center=(0, 0, z), radius=5, n_sides=100, fill=False) for z in np.linspace(-8, 8, 16)
]
coil_block = pv.MultiBlock(coils)
coil_block.plot(render_lines_as_tubes=True, line_width=10)


# %%
# Compute and Plot Field Lines
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Next, let's compute streamlines from the center of the coil to represent the
# direction of the magnetic force. For this, we can create a simple
# :func:`pyvista.Disc` and use that as the source of the streamlines.

seed = pv.Disc(inner=1, outer=5.4, r_res=2, c_res=12)
strl = grid.streamlines_from_source(
    seed,
    vectors='B',
    max_length=180,
    initial_step_length=0.1,
    integration_direction='both',
)

pl = pv.Plotter()
pl.add_mesh(
    strl.tube(radius=0.1),
    cmap='bwr',
    ambient=0.2,
)
pl.add_mesh(coil_block, render_lines_as_tubes=True, line_width=5, color='w')
pl.camera.zoom(3)
pl.show()


# %%
# Plot the Magnet Field Strength
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Finally, let's bring this all together by plotting the magnetic field
# strength while also plotting the streamlines and the coil.

# Take the norm of the magnetic field
scalars = np.linalg.norm(grid['B'], axis=1)

# Customize the opacity to make it easier to visualize the strength of the
# field nearby the coil
opacity = [
    0.00,
    0.05,
    0.1,
    0.3,
    0.3,
    0.5,
    0.95,
    0.95,
    1.0,
    1.0,
]

# sphinx_gallery_start_ignore
# volume rendering does not work in interactive plots currently
PYVISTA_GALLERY_FORCE_STATIC = True
# sphinx_gallery_end_ignore

# Add this all to the plotter
pl = pv.Plotter()
pl.add_mesh(
    strl.tube(radius=0.1),
    color='black',
)
pl.add_mesh(coil_block, render_lines_as_tubes=True, line_width=5, color='w')
vol = pl.add_volume(
    grid,
    scalars=scalars,
    opacity=opacity,
    cmap='hot',
    show_scalar_bar=False,
)
vol.prop.interpolation_type = 'linear'
pl.camera.zoom(5)
pl.show()
