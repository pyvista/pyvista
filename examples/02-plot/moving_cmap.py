"""
.. _moving_cmap_example:

Create a GIF Movie of a Static Object with a Moving Colormap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generate a gif movie of a Hopf torus with a moving colormap,
by updating the scalars.
This example uses :meth:`~pyvista.Plotter.open_gif` and
:meth:`~pyvista.Plotter.write_frame` to create the gif.

"""

from __future__ import annotations

import numpy as np

import pyvista as pv


# A spherical curve
def scurve(t):
    alpha = np.pi / 2 - (np.pi / 2 - 0.44) * np.cos(3 * t)
    beta = t + 0.44 * np.sin(6 * t)
    return np.array([np.sin(alpha) * np.cos(beta), np.sin(alpha) * np.sin(beta), np.cos(alpha)])


# Hopf fiber
def hopf_fiber(p, phi):
    return np.array(
        [
            (1 + p[2]) * np.cos(phi),
            p[0] * np.sin(phi) - p[1] * np.cos(phi),
            p[0] * np.cos(phi) + p[1] * np.sin(phi),
            (1 + p[2]) * np.sin(phi),
        ],
    ) / np.sqrt(2 * (1 + p[2]))


# Stereographic projection
def stereo_proj(q):
    return q[0:3] / (1 - q[3])


# Parameterization of the Hopf torus
def hopf_torus(t, phi):
    return stereo_proj(hopf_fiber(scurve(t), phi))


# Create the mesh
angle_u = np.linspace(-np.pi, np.pi, 400)
angle_v = np.linspace(0, np.pi, 200)
u, v = np.meshgrid(angle_u, angle_v)
x, y, z = hopf_torus(u, v)
grid = pv.StructuredGrid(x, y, z)
mesh = grid.extract_surface().clean(tolerance=1e-6)

# Distances normalized to [0, 2*pi]
dists = np.linalg.norm(mesh.points, axis=1)
dists = 2 * np.pi * (dists - dists.min()) / (dists.max() - dists.min())

mesh['distances'] = np.sin(dists)

# Make the movie
pltr = pv.Plotter(window_size=[512, 512])
pltr.set_focus([0, 0, 0])
pltr.set_position([40, 0, 0])
pltr.add_mesh(
    mesh,
    scalars='distances',
    smooth_shading=True,
    specular=1,
    cmap='nipy_spectral',
    show_scalar_bar=False,
)
pltr.open_gif('Hopf_torus.gif')

for t in np.linspace(0, 2 * np.pi, 60, endpoint=False):
    mesh['distances'] = np.sin(dists - t)
    pltr.write_frame()

pltr.show()
# %%
# .. tags:: plot
