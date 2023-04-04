""".. _cfd_openfoam_example:

Plot CFD Data
~~~~~~~~~~~~~
Plot a CFD example from OpenFoam.

See :ref:`openfoam_example` for a full example using :class:`pyvista.POpenFOAMReader`.

"""

import numpy as np

import pyvista as pv

reader = pv.OpenFOAMReader('./case.foam')
reader.set_active_time_value(1000)
mesh = reader.read()
# mesh[0].slice_along_axis(7, axis='z').plot(scalars='k', preference='point')

# lines, src = mesh[0].streamlines(vectors='U', source_center=(0, 0, 0), source_radius=0.02, return_source=True, max_time=0.5, integration_direction='backward', n_points=50)

# pl = pv.Plotter()
# pl.add_mesh(lines, render_lines_as_tubes=True, line_width=1, lighting=False, scalars='U')
# # pl.add_points(src)
# pl.add_mesh(mesh[1], color='w', opacity=0.05)
# pl.enable_anti_aliasing()
# pl.camera_position='xz'
# pl.show()


bounds = np.array(mesh[1].bounds) * 1.2
origin = (bounds[0], bounds[2], bounds[4])

sp = 0.004
spacing = (sp, sp, sp)

dimensions = (
    int((bounds[1] - bounds[0]) // sp + 2),
    int((bounds[3] - bounds[2]) // sp + 2),
    int((bounds[5] - bounds[4]) // sp + 2),
)

grid = pv.UniformGrid(dimensions=dimensions, spacing=spacing, origin=origin)

flow = mesh[0].copy()
flow.clear_data()
# flow['u-norm'] = np.linalg.norm(mesh[0].point_data['U'], axis=1)
flow['p'] = mesh[0].point_data['nut']

out = grid.interpolate(flow, radius=0.005)
pl = pv.Plotter()
vol = pl.add_volume(out, opacity='linear')
# , opacity=[0, 0, 0.0, 0.0, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4])
# vol.mapper.blend_mode = 'maximum'
vol.prop.interpolation_type = 'linear'
pl.add_mesh(mesh[1], color='w', opacity=0.1)
pl.camera_position = 'xz'
pl.show()
