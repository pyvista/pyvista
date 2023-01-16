r"""
.. _plot_atomic_orbitals_example:

Plot Atomic Orbitals
--------------------
Visualize the wave functions (orbitals) of the hydrogen atom.

"""

###############################################################################
# Import
# ~~~~~~
# Import the applicable libraries.
#
# .. note::
#    This example is modeled off of `Matplotlib: Hydrogen Wave Function
#    <http://staff.ustc.edu.cn/~zqj/posts/Hydrogen-Wavefunction/>`_
#
#    This example requires `sympy <https://www.sympy.org/>`_. Install it with:
#
#    .. code:: python
#       pip install sympy

import numpy as np

import pyvista as pv
from pyvista import examples

###############################################################################
# Generate the Dataset
# ~~~~~~~~~~~~~~~~~~~~
# Generate the dataset by evaluating the analytic hydrogen wave function from
# ``sympy``.
#
# .. math::
#    \begin{equation}
#        \label{eq:hydro_wfc}
#        \psi_{nlm}(r,\theta,\phi)
#        =
#        \sqrt{
#            \left(\frac{2}{na_0}\right)^3\, \frac{(n-l-1)!}{2n[(n+l)!]}
#        }
#        e^{-r / na_0}
#        \left(\frac{2r}{na_0}\right)^l
#        L_{n-l-1}^{2l+1} \cdot Y_l^m(\theta, \phi)
#    \end{equation}
#
# See `Hydrogen atom <https://en.wikipedia.org/wiki/Hydrogen_atom>`_ for more
# details.
#
# This dataset evaluates this function for the hydrogen orbital 3dxy, with the
# following quantum numbers:
#
# * Principal quantum number: ``n=3``
# * Azimuthal quantum number: ``l=2``
# * Magnetic quantum number: ``m=-2``

grid = examples.load_hydrogen_orbital(3, 2, -2)
grid


###############################################################################
# Plot the Orbital
# ~~~~~~~~~~~~~~~~
# Plot the orbital using :func:`add_volume() <pyvista.Plotter.add_volume>` and
# using the default scalars contained in ``grid``, ``real_hwf``. This way we
# can plot more than just the probability of the election, but also the phase
# of the electron.
#
# .. note::
#    Since the real value of evaluated wave function for this orbital varies
#    between ``[-<value>, <value>]``, we cannot use the default opacity
#    ``opacity='linear'``. Instead, we use ``[1, 0, 1]`` since we would like
#    the opacity to be proportional to the absolute value of the scalars.

pl = pv.Plotter()
vol = pl.add_volume(grid, cmap='magma', opacity=[1, 0, 1])
vol.prop.interpolation_type = 'linear'
pl.camera.zoom(1.5)
pl.background_color = [0.2, 0.2, 0.2]
pl.show_axes()
pl.show()


###############################################################################
# Plot the Orbital Contours
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Generate the contour plot for the orbital by determining when the orbital
# equals 10% the maximum value of the orbital. This effectively captures the
# majority of the "volume" the electron potentially exists in.
#
# Note how we use the absolute value of the scalars when evaluating
# :func:`contour() <pyvista.PolyDataFilters.contour>`. Otherwise, we would only
# capture the location of the spin-up electron.

eval_at = grid['real_hwf'].max() * 0.1
contours = grid.contour(
    [eval_at],
    scalars=np.abs(grid['real_hwf']),
    method='marching_cubes',
)
contours = contours.interpolate(grid)
contours.plot(
    categories=2,
    smooth_shading=True,
    annotations={-eval_at: 'down-spin', eval_at: 'up-spin'},
    scalar_bar_args={'n_labels': 0, 'title': ''},
)


###############################################################################
# Plot all the

hydro_orbital = examples.load_hydrogen_orbital(3, 0, 0)


def plot_orbital(orbital):
    # normalize opacity
    # scalars = np.vstack((orbital['real_hwf'], np.abs(orbital['real_hwf']))).T
    # scalars = np.abs(orbital['real_hwf'])

    neg_mask = orbital['real_hwf'] < 0
    arr = np.zeros((orbital.n_points, 4), np.uint8)
    arr[neg_mask, 0] = 255
    arr[~neg_mask, 1] = 255

    # normalize opacity
    opac = np.abs(orbital['real_hwf'])
    opac /= opac.max()
    arr[:, -1] = opac * 255

    orbital['plot_scalars'] = arr

    pl = pv.Plotter()
    vol = pl.add_volume(
        orbital,
        scalars='plot_scalars',
    )
    vol.prop.interpolation_type = 'linear'
    pl.add_volume_clipper(vol, normal='-x')
    pl.camera.zoom(1.5)
    pl.background_color = [0.2, 0.2, 0.2]
    pl.show_axes()
    pl.show()


plot_orbital(hydro_orbital)
