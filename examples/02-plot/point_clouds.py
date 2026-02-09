"""
.. _point_clouds_example:

Plotting Point Clouds
~~~~~~~~~~~~~~~~~~~~~
This example shows you how to plot point clouds using PyVista using both the
``'points'`` and ``'points_gaussian'`` styles.

"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvista import examples

# sphinx_gallery_start_ignore
# point gaussian does not work in interactive plots
PYVISTA_GALLERY_FORCE_STATIC_IN_DOCUMENT = True
# sphinx_gallery_end_ignore

# %%
# Compare the Plotting methods
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# First, let's create a sample point cloud using :func:`numpy.random.random`.

# Seed rng for reproducibility
rng = np.random.default_rng(seed=0)
points = rng.random((1000, 3))
points

# %%
# Basic Plot
# ~~~~~~~~~~
# We can simply plot this point cloud using the convenience :func:`pyvista.plot`
# function.

pv.plot(points)


# %%
# Plot with Scalars
# ~~~~~~~~~~~~~~~~~
# That's quite boring, so let's spice things up by adding color. We can either
# use a single scalar to plot the points. For example, the z coordinates.
#
# For fun, let's also render the points as spheres.
pv.plot(
    points,
    scalars=points[:, 2],
    render_points_as_spheres=True,
    point_size=20,
    show_scalar_bar=False,
)


# %%
# Plot with RGBA
# ~~~~~~~~~~~~~~
# Alternatively, we can color the point cloud using an RGBA array. This has
# been normalized from (0, 1), but we could have also used a ``numpy.uint8``
# array from 0-255.
rgba = points - points.min(axis=0)
rgba /= rgba.max(axis=0)
pv.plot(
    points,
    scalars=rgba,
    render_points_as_spheres=True,
    point_size=20,
    cpos='xy',
    rgba=True,
)

# %%
# Point Cloud Plot Styles
# ~~~~~~~~~~~~~~~~~~~~~~~
# PyVista supports the ``'points_gaussian'`` style, which renders points as
# individual soft sprites. You have the option of displaying these as tight
# "spheres" using ``render_points_as_spheres=True`` (default), or disabling it
# to create softer points at the expense of render performance.
#
# Here's the basic plot again, but with the style as ``'points_gaussian'``:
pv.plot(points, style='points_gaussian', opacity=0.5, point_size=15)


# %%
# Here's a plotter with four combinations of the options side-by-side so you
# can see for yourself the different options available when plotting these
# points. PyVista tries to achieve sensible defaults, but should you find these
# insufficient for your needs, feel free to play around with the various options
# and find something that works for you.

pl = pv.Plotter(shape=(2, 2))

# Standard points
actor = pl.add_points(
    points,
    style='points',
    emissive=False,
    scalars=rgba,
    rgba=True,
    point_size=10,
    ambient=0.7,
)
pl.add_text('"points" not as spheres', color='w')

# Gaussian points
pl.subplot(0, 1)
actor = pl.add_points(
    points,
    render_points_as_spheres=False,
    style='points_gaussian',
    emissive=False,
    scalars=rgba,
    rgba=True,
    opacity=0.99,
    point_size=10,
    ambient=1.0,
)
pl.add_text('"points_gaussian" not as spheres\nemissive=False', color='w')

# Gaussian points with emissive=True
pl.subplot(1, 0)
actor = pl.add_points(
    points,
    render_points_as_spheres=False,
    style='points_gaussian',
    emissive=True,
    scalars=rgba,
    rgba=True,
    point_size=10,
)
pl.add_text('"points_gaussian" not as spheres\nemissive=True', color='w')

# With render_points_as_spheres=True
pl.subplot(1, 1)
actor = pl.add_points(
    points,
    style='points_gaussian',
    render_points_as_spheres=True,
    scalars=rgba,
    rgba=True,
    point_size=10,
)
pl.add_text('"points_gaussian" as spheres', color='w')

pl.background_color = 'k'
pl.link_views()
pl.camera_position = 'xy'
pl.camera.zoom(1.2)
pl.show()


# %%
# Orbit a Point Cloud
# ~~~~~~~~~~~~~~~~~~~
# Generate a plot orbiting around a point cloud. Color based on the distance
# from the center of the cloud using :func:`~pyvista.Plotter.generate_orbital_path`.

cloud = examples.download_cloud_dark_matter()
scalars = np.linalg.norm(cloud.points - cloud.center, axis=1)

pl = pv.Plotter(off_screen=True)
pl.add_mesh(
    cloud,
    style='points_gaussian',
    color='#fff7c2',
    scalars=scalars,
    opacity=0.25,
    point_size=4.0,
    show_scalar_bar=False,
)
pl.background_color = 'k'
pl.show(auto_close=False)
path = pl.generate_orbital_path(n_points=36, shift=cloud.length, factor=3.0)
pl.open_gif('orbit_cloud.gif')
pl.orbit_on_path(path, write_frames=True)
pl.close()
# %%
# .. tags:: plot
