"""
.. _moving_cmap_example:

Create a GIF Movie of a Static Object with a Moving Colormap
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generate a gif movie of a Hopf torus with a moving colormap, 
by updating the scalars.

"""

from math import pi

import numpy as np

import pyvista as pv

# A spherica curve
def Gamma(t):
    alpha = pi / 2 - (pi / 2 - 0.44) * np.cos(3 * t)
    beta = t + 0.44 * np.sin(6 * t)
    return np.array(
        [
          np.sin(alpha) * np.cos(beta), 
          np.sin(alpha) * np.sin(beta), 
          np.cos(alpha)
        ]
    )

# Hopf fiber
def HopfFiber(p, phi):
    return (
        np.array(
            [
                (1 + p[2]) * np.cos(phi),
                p[0] * np.sin(phi) - p[1] * np.cos(phi),
                p[0] * np.cos(phi) + p[1] * np.sin(phi),
                (1 + p[2]) * np.sin(phi),
            ]
        )
        / np.sqrt(2 * (1 + p[2]))
    )

# Stereographic projection
def Stereo(q):
    return q[0:3] / (1 - q[3])

# Parameterization of the Hopf torus
def F(t, phi):
    return Stereo(HopfFiber(Gamma(t), phi))

# Create the mesh
angle_u = np.linspace(-pi, pi, 400)
angle_v = np.linspace(0, pi, 200)
u, v = np.meshgrid(angle_u, angle_v)
x, y, z = F(u, v)
grid = pv.StructuredGrid(x, y, z)
mesh = grid.extract_geometry().clean(tolerance=1e-6)

# Distances normalized to [0, 2*pi]
dists = np.linalg.norm(mesh.points, axis=1)
dists = 2 * pi * (dists - dists.min()) / (dists.max() - dists.min())

# Make the movie
pltr = pv.Plotter(window_size=[512,512])
pltr.set_focus([0, 0, 0])
pltr.set_position((40, 0, 0))
pltr.add_mesh(
    mesh,
    scalars=np.sin(dists),
    smooth_shading=True,
    specular=10,
    cmap="flag",
    show_scalar_bar=False,
)
pltr.open_gif("Hopf_torus.gif")

for t in np.linspace(0, 2 * pi, 60, endpoint=False):
    pltr.update_scalars(np.sin(dists - t), render=False)
    pltr.write_frame()

pltr.show()
