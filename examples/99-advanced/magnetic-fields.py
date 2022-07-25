"""
.. _magnetic_fields_example:

Plot Electric and Magnetic Fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following example demonstrates how PyVista can plot electric and magnetic
fields.

The modeling results are courtesy of `prisae <https://github.com/prisae>`_ and
are from the `emg3d Minimum working example
<https://emsig.xyz/emg3d-gallery/gallery/tutorials/minimum_example.html#sphx-glr-gallery-tutorials-minimum-example-py>`_.

"""
# sphinx_gallery_thumbnail_number = 2

import numpy as np

import pyvista as pv
from pyvista import examples


###############################################################################
# Let's first download the example dataset and show that it's a
# :class:`pyvista.UniformGrid` with the electric field encoded as three arrays within the ``point_data``
grid = examples.download_dipole_efield()
grid.point_data

efield = np.vstack((grid['efield_fx'], grid['efield_fy'], grid['efield_fz'])).T
grid['efield'] = efield

###############################################################################
# First, let's take a slice of the electric field in the X in the XY plane.

field_slice = grid.slice('z')
field_slice.plot(
    scalars='efield_fx',
    cpos='xy',
    rng=[1E-15, 1E-5],
    component=0,
    log_scale=True,
    lighting=False,
)

###############################################################################

field_slice.glyph(geom=pv.Arrow(), factor=1E5).plot(cpos='xy')


###############################################################################
# Next, lets combine the individual directions into a 3D component and then
# plot the normalized electric field in the XY plane


field_slice = grid.slice('z')
field_slice.plot(
    scalars='efield',
    cpos='xy',
    rng=[1E-15, 1E-4],
    log_scale=True,
    lighting=False,
)

###############################################################################
#

hfield_grid = examples.download_dipole_hfield()

hfield = np.vstack((hfield_grid['hfield_fx'], hfield_grid['hfield_fy'], hfield_grid['hfield_fz'])).T

hfield_grid['hfield'] = hfield

hfield_grid.streamlines(source_center=(c, c, c)).plot()


# hfield_grid.glyph(geom=pv.Arrow())
