"""
.. _openfoam_tubes_example:

Plot CFD Data
-------------
Plot a CFD example from OpenFoam hosted on the public SimScale examples at
`SimScale Project Library <https://www.simscale.com/projects/>`_.

This example dataset was read using the :class:`pyvista.POpenFOAMReader`. See
:ref:`openfoam_example` for a full example using this reader.

"""

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
# Download and load the example dataset.

block = examples.download_openfoam_tubes()
block


# %%
# Plot Cross Section
# ~~~~~~~~~~~~~~~~~~
# Plot the outline of the dataset along with a cross section of the flow velocity.

# first, get the first block representing the air within the tube.
air = block[0]

# generate a slice in the XZ plane
y_slice = air.slice('y')

pl = pv.Plotter()
pl.add_mesh(y_slice, scalars='U', lighting=False, scalar_bar_args={'title': 'Flow Velocity'})
pl.add_mesh(air, color='w', opacity=0.25)
pl.enable_anti_aliasing()
pl.show()


# %%
# Plot Streamlines - Flow Velocity
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Generate streamlines using :func:`streamlines_from_source()
# <pyvista.DataSetFilters.streamlines_from_source>`.

# Let's use the inlet as a source. First plot it.
inlet = block[1][2]
pl = pv.Plotter()
pl.add_mesh(inlet, color='b', label='inlet')
pl.add_mesh(air, opacity=0.2, color='w', label='air')
pl.enable_anti_aliasing()
pl.add_legend(face=None)
pl.show()


# %%
# Now, actually generate the streamlines. Since the original inlet contains
# 1000 points, let's reduce this to around 200 points by using every 5th point.
#
# .. note::
#    If we wanted a uniform subsampling of the inlet, we could use
#    `pyvista/pyacvd <https://github.com/pyvista/pyacvd>`_

pset = pv.PointSet(inlet.points[::5])
lines = air.streamlines_from_source(
    pset,
    vectors='U',
    max_length=1.0,
)

pl = pv.Plotter()
pl.add_mesh(
    lines,
    render_lines_as_tubes=True,
    line_width=3,
    lighting=False,
    scalar_bar_args={'title': 'Flow Velocity'},
    scalars='U',
    rng=(0, 212),
)
pl.add_mesh(air, color='w', opacity=0.25)
pl.enable_anti_aliasing()
pl.camera_position = 'xz'
pl.show()


# %%
# Volumetric Plot - Visualize Turbulent Kinematic Viscosity
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The turbulent kinematic viscosity of a fluid is a derived quantity used in
# turbulence modeling to describe the effect of turbulent motion on the
# momentum transport within the fluid.
#
# For this example, we will first sample the results from the
# :class:`pyvista.UnstructuredGrid` onto a :class:`pyvista.ImageData` using
# :func:`sample() <pyvista.DataObjectFilters.sample>`. This is so we can visualize
# it using :func:`add_volume() <pyvista.Plotter.add_volume>`

# sphinx_gallery_start_ignore
# volume rendering does not work in interactive plots currently
PYVISTA_GALLERY_FORCE_STATIC = True
# sphinx_gallery_end_ignore

bounds = np.array(air.bounds) * 1.2
origin = (bounds[0], bounds[2], bounds[4])
spacing = (0.003, 0.003, 0.003)
dimensions = (
    int((bounds[1] - bounds[0]) // spacing[0] + 2),
    int((bounds[3] - bounds[2]) // spacing[1] + 2),
    int((bounds[5] - bounds[4]) // spacing[2] + 2),
)
grid = pv.ImageData(dimensions=dimensions, spacing=spacing, origin=origin)
grid = grid.sample(air)

pl = pv.Plotter()
vol = pl.add_volume(
    grid,
    scalars='nut',
    opacity='linear',
    scalar_bar_args={'title': 'Turbulent Kinematic Viscosity'},
)
vol.prop.interpolation_type = 'linear'
pl.add_mesh(air, color='w', opacity=0.1)
pl.camera_position = 'xz'
pl.show()
