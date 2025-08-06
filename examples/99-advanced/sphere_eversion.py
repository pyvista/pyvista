"""
.. _sphere_eversion_example:

Turning the sphere inside out
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are several videos online talking about how a sphere can be turned inside
out in a continuous fashion, for instance in `this YouTube video
<https://www.youtube.com/watch?v=OI-To1eUtuU>`_.  Thanks to `an excellent paper
by Adam Bednorz and Witold Bednorz, Differential and its Applications 64, 59
(2019) <https://doi.org/10.1016/j.difgeo.2019.02.004>`_ (also available `on
arXiv <https://arxiv.org/abs/1711.10466>`_), we can plot this so-called
eversion of a sphere (turning it inside out without pinching or tearing the
surface, in other words by preserving its topology).

The mathematics involved can seem a bit, well, involved. What matters is the
overall process visible in the animation: first the sphere is corrugated and
stretched out a bit to allow some legroom for the smooth transformation, then
the lobes are twisted around through each other, and the process is reversed in
order to unfold the sphere. It's not obvious that the transformation is truly
smooth; this was proved in the paper by Bednorz and Bednorz.

"""

# sphinx_gallery_thumbnail_number = 2
from __future__ import annotations

import numpy as np

import pyvista as pv

# define some parameters
n_steps = 30  # number of steps for a given stage of the animation
Q = 2 / 3  # arbitrary < 1
w = 2  # arbitrary > 0
n = 2  # arbitrary integer > 1, the number of "lobes"
beta = 1  # arbitrary > 1
alpha_final = 1  # arbitrary > 1
eta_final = 2  # arbitrary > 1
kappa = (n - 1) / (2 * n)

# %%
# Let's define the chain of mappings we'll need for implementing the eversion:


def sphere_to_cylinder(theta, phi):
    """Map from spherical polar coordinates to cylindrical ones.

    Input theta goes from -pi/2 to pi/2, phi goes from -pi to pi.
    Output h goes from -infinity to infinity, phi goes from -phi to phi.

    """
    h = w * np.sin(theta) / np.cos(theta) ** n
    # phi is unchanged
    return h, phi


def cylinder_to_wormhole(h, phi, t, p, q):  # noqa: PLR0917
    """
    Map from a cylinder to an open wormhole using Eq. (4).

    Input h goes from -infinity to infinity, phi goes from -phi to phi.
    Output is an (x, y, z) point embedded in 3d space.

    The parameters t, p, q vary during the eversion process.
    Start from |t| > 1 (fixed), p = 1 and q = 0. End at p = 0, qt = +-1.

    """
    x = t * np.cos(phi) + p * np.sin((n - 1) * phi) - h * np.sin(phi)
    y = t * np.sin(phi) + p * np.cos((n - 1) * phi) + h * np.cos(phi)
    z = h * np.sin(n * phi) - t / n * np.cos(n * phi) - q * t * h
    return x, y, z


def close_wormhole(x0, y0, z0, eta, xi, alpha):  # noqa: PLR0917
    """
    Close the wormhole using Eqs. (7)-(8).

    Input is an (x0, y0, z0) point embedded in 3d space.
    Output is an (x2, y2, z2) == (x'', y'', z'') point embedded in 3d space.

    The parameters eta, xi, alpha vary during the eversion process.

    """
    # Eq. (7): (x, y, z) -> (x', y', z')
    denominator = xi + eta * (x**2 + y**2)
    x1 = x0 / (denominator**kappa)
    y1 = y0 / (denominator**kappa)
    z1 = z0 / denominator

    gamma = 2 * np.sqrt(alpha * beta)
    # singular case, Eq (9):
    if np.isclose(gamma, 0):
        denominator = x1**2 + y1**2
        x2 = x1 / denominator
        y2 = y1 / denominator
        z2 = -z1
        return x2, y2, z2

    # Eq. (8): (x', y', z') -> (x'', y'', z'')
    exponential = np.exp(gamma * z1)
    numerator = alpha - beta * (x1**2 + y1**2)
    denominator = alpha + beta * (x1**2 + y1**2)
    x2 = x1 * exponential / denominator
    y2 = y1 * exponential / denominator
    z2 = numerator / denominator * exponential / gamma - (alpha - beta) / (alpha + beta) / gamma
    return x2, y2, z2


def unfold_sphere(theta, phi, t, q, eta, lamda):  # noqa: PLR0917
    """
    Unfold the sphere using Eqs. (12), (15), (10).

    Input is a (theta, phi) point in spherical coordinates.
    Output is an (x, y, z) point embedded in 3d space.

    The parameter lamda varies. Lamda = 1 is the final stage of the
    wormhole closing, and lamda = 0 is the recovered sphere.

    """
    # apply Eqs. (12), (15)
    # fmt: off
    x = (
        t * (1 - lamda + lamda * np.cos(theta)**n) * np.cos(phi)
        - lamda * w * np.sin(theta) * np.sin(phi)
    )
    x /= np.cos(theta)**n
    y = (
        t * (1 - lamda + lamda * np.cos(theta)**n) * np.sin(phi)
        + lamda * w * np.sin(theta) * np.cos(phi)
    )
    y /= np.cos(theta) ** n
    z = (
        lamda * (
            (w * np.sin(theta) * (np.sin(n * phi) - q * t)) / np.cos(theta)**n
            - t / n * np.cos(n * phi)
        )
        - (1 - lamda) * eta**(1 + kappa) * t * abs(t)**(2 * kappa)
            * np.sin(theta) / np.cos(theta)**(2 * n)
    )
    # fmt: on

    # apply Eq. (10)
    denominator = x**2 + y**2
    x2 = x * eta**kappa / denominator ** (1 - kappa)
    y2 = y * eta**kappa / denominator ** (1 - kappa)
    z2 = -z / eta / denominator
    return x2, y2, z2


# %%
# Now chain the functions by performing the process in Table 1 of the paper.
# Start from the bottom for ``t = -1/Q``, keep stepping up, linearly changing
# parameters that change from row to row, then at the top go from ``t = -1/Q``
# to ``t = 1/Q``, then go back from top to bottom. Save each frame to a GIF.
#
# We make good use of the ``backface_params`` keyword parameter of
# :func:`pyvista.Plotter.add_mesh`, allowing us to plot the inside and the
# outside with different colors.

# plot options to use for each frame
opts = dict(
    color='aquamarine',
    specular=1.0,
    specular_power=50.0,
    backface_params=dict(color='forestgreen'),
    smooth_shading=True,
    reset_camera=True,
)

# use a small figure window to reduce the size of the GIF
plotter = pv.Plotter(window_size=(300, 300))
plotter.open_gif('sphere_eversion.gif')


def save_frame(x, y, z):
    """Generate and store a frame of the eversion."""
    plotter.clear()
    plotter.enable_lightkit()
    plotter.add_mesh(pv.StructuredGrid(x, y, z), **opts)
    plotter.write_frame()


# initial parameters, will be updated
t = -1 / Q
q = Q
p = xi = alpha = 0
eta = 1

# sphere -> inverted wormhole
theta, phi = np.mgrid[-np.pi / 2 : np.pi / 2 : 200j, -np.pi : np.pi : 400j]
h, phi = sphere_to_cylinder(theta, phi)
for lamda in np.linspace(0, 1, n_steps, endpoint=False):
    x2, y2, z2 = unfold_sphere(theta, phi, t, q, eta, lamda)
    save_frame(x2, y2, z2)

# inverted wormhole -> unfolded wormhole
x, y, z = cylinder_to_wormhole(h, phi, t, p, q)
xis = np.linspace(0, 1, n_steps)
alphas = np.linspace(0, alpha_final, n_steps)
etas = np.linspace(1, eta_final, n_steps)
for xi, alpha, eta in zip(xis, alphas, etas):
    x2, y2, z2 = close_wormhole(x, y, z, eta, xi, alpha)
    save_frame(x2, y2, z2)

# unfolded wormhole -> closed wormhole
for q in np.linspace(Q, 0, n_steps):
    p = 1 - abs(q * t)
    x, y, z = cylinder_to_wormhole(h, phi, t, p, q)
    x2, y2, z2 = close_wormhole(x, y, z, eta, xi, alpha)
    save_frame(x2, y2, z2)

# closed wormhole turned inside out (flip sign of time)
# unfolded wormhole -> closed wormhole
for t in np.linspace(-1 / Q, 1 / Q, n_steps):
    p = 1 - abs(q * t)
    x, y, z = cylinder_to_wormhole(h, phi, t, p, q)
    x2, y2, z2 = close_wormhole(x, y, z, eta, xi, alpha)
    save_frame(x2, y2, z2)

# closed wormhole -> unfolded wormhole
for q in np.linspace(0, Q, n_steps + 1)[1:]:
    p = 1 - abs(q * t)
    x, y, z = cylinder_to_wormhole(h, phi, t, p, q)
    x2, y2, z2 = close_wormhole(x, y, z, eta, xi, alpha)
    save_frame(x2, y2, z2)

# unfolded wormhole -> inverted wormhole
x, y, z = cylinder_to_wormhole(h, phi, t, p, q)
xis = np.linspace(1, 0, n_steps + 1)[1:]
alphas = np.linspace(alpha_final, 0, n_steps + 1)[1:]
etas = np.linspace(eta_final, 1, n_steps + 1)[1:]
for xi, alpha in zip(xis, alphas):
    x2, y2, z2 = close_wormhole(x, y, z, eta, xi, alpha)
    save_frame(x2, y2, z2)

# inverted wormhole -> sphere
for lamda in np.linspace(1, 0, n_steps + 1)[1:]:
    x2, y2, z2 = unfold_sphere(theta, phi, t, q, eta, lamda)
    save_frame(x2, y2, z2)

plotter.close()


# %%
# Looking at the still image of the middle state with ``t = 0``, we see a nice
# symmetric configuration where two "inside" and two "outside" lobes of the
# sphere are visible.

# sphinx_gallery_start_ignore
# lighting does not work for this interactive plot
PYVISTA_GALLERY_FORCE_STATIC = True
# sphinx_gallery_end_ignore

t = q = 0
xi = p = 1
eta = eta_final
alpha = alpha_final

x, y, z = cylinder_to_wormhole(h, phi, t, p, q)
x2, y2, z2 = close_wormhole(x, y, z, eta, xi, alpha)

plotter = pv.Plotter(window_size=(512, 512))
plotter.add_mesh(pv.StructuredGrid(x2, y2, z2), **opts)
plotter.show()
