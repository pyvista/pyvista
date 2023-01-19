"""
.. _magnetic_fields_example:

Plot Electric and Magnetic Fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following example demonstrates how PyVista can plot magnetic fields.

"""
# sphinx_gallery_thumbnail_number = 2

import numpy as np

import pyvista as pv
from pyvista import examples

###############################################################################
# Let's first download the example dataset and show that it's a
# :class:`pyvista.UniformGrid` with the electric field encoded as three arrays within the ``point_data``
grid = examples.download_coil_magnetic_field()
grid.point_data


# compute field lines
seed = pv.Disc(inner=1, outer=5.2, r_res=3, c_res=12)
strl = grid.streamlines_from_source(
    seed,
    vectors='B',
    max_time=180,
    initial_step_length=0.1,
    integration_direction='both',
)


# create coils
coils = []
for z in np.linspace(-8, 8, 16):
    coils.append(pv.Polygon((0, 0, z), radius=5, n_sides=100, fill=False))
coils = pv.MultiBlock(coils)
# plot the magnet field strength in the Z direction

scalars = np.linalg.norm(grid['B'], axis=1)
opacity = 1 - np.geomspace(1, 1e-5, 10)

scalars = np.abs(grid['B'][:, 2])
pl = pv.Plotter()

pl.add_mesh(
    strl.tube(radius=0.1),
    # cmap="bwr",
    color='black',
    ambient=0.2,
)

pl.add_mesh(coils, render_lines_as_tubes=True, line_width=5, color='w')
vol = pl.add_volume(
    grid, scalars=scalars, opacity=opacity, cmap='bwr', show_scalar_bar=False, log_scale=True
)
pl.add_volume_clip_plane(
    vol,
    normal='-x',
    normal_rotation=False,
    interaction_event='always',
    widget_color=pv.Color(opacity=0.0),
)
pl.enable_anti_aliasing()
pl.camera.zoom(2)
pl.background_color = [0.2, 0.2, 0.2]
pl.show()
