""".. _openfoam_tubes_example:

Plot CFD Data
-------------
Plot a CFD example from OpenFoam hosted on the public SimScale examples at
`SimScale Project Library <https://www.simscale.com/projects/>`_.

This example dataset was read using the :class:`pyvista.POpenFOAMReader`. See
:ref:`openfoam_example` for a full example using this reader.

"""

import numpy as np

import pyvista as pv
from pyvista import examples

###############################################################################
# Download and load the example dataset.

mesh = examples.download_openfoam_tubes()
mesh


###############################################################################
# Plot Cross Section
# ~~~~~~~~~~~~~~~~~~
# Plot the outline of the dataset along with a cross section of the flow velocity.

# generate a slice in the XZ plane
y_slice = mesh.slice('y')

pl = pv.Plotter()
pl.add_mesh(y_slice, scalars='U', lighting=False, scalar_bar_args={'title': 'Flow Velocity'})
pl.add_mesh(mesh, color='w', opacity=0.25)
pl.enable_anti_aliasing()
pl.show()


###############################################################################
# Plot Streamlines - Flow Velocity
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Generate streamlines using :func:`streamlines() <pyvista.DataSetFilters.streamlines>`.

lines, src = mesh.streamlines(
    vectors='U',
    source_center=(0, 0, 0),
    source_radius=0.025,
    return_source=True,
    max_time=0.5,
    integration_direction='backward',
    n_points=40,
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
pl.add_mesh(mesh, color='w', opacity=0.25)
pl.enable_anti_aliasing()
pl.camera_position = 'xz'
pl.show()


###############################################################################
# Volumetric Plot - Visualize Turbulent Kinematic Viscosity
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The turbulent kinematic viscosity of a fluid is a derived quantity used in
# turbulence modeling to describe the effect of turbulent motion on the
# momentum transport within the fluid.
#
# For this example, we will first interpolate the results from the
# :class:`pyvista.UnstructuredGrid` onto a :class:`pyvista.UniformGrid` using
# :func:`interpolate() <pyvista.DataSetFilters.interpolate>`. This is so we can
# visualize it using :func:`add_volume() <pyvista.Plotter.add_volume>`

bounds = np.array(mesh.bounds) * 1.2
origin = (bounds[0], bounds[2], bounds[4])
spacing = (0.003, 0.003, 0.003)
dimensions = (
    int((bounds[1] - bounds[0]) // spacing[0] + 2),
    int((bounds[3] - bounds[2]) // spacing[1] + 2),
    int((bounds[5] - bounds[4]) // spacing[2] + 2),
)
grid = pv.UniformGrid(dimensions=dimensions, spacing=spacing, origin=origin)
grid = grid.interpolate(mesh, radius=0.005)

pl = pv.Plotter()
vol = pl.add_volume(
    grid,
    scalars='nut',
    opacity='linear',
    scalar_bar_args={'title': 'Turbulent Kinematic Viscosity'},
)
vol.prop.interpolation_type = 'linear'
pl.add_mesh(mesh, color='w', opacity=0.1)
pl.camera_position = 'xz'
pl.show()
