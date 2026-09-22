"""
.. _movie_glyphs_example:

Save a Movie Using Glyphs
~~~~~~~~~~~~~~~~~~~~~~~~~

Create an animated GIF by generating glyphs from a scalar field.

Uses :func:`glyph() <pyvista.DataSetFilters.glyph>` with
:func:`pyvista.Sphere`.

"""

# sphinx_gallery_thumbnail_number = 1
import numpy as np
import pyvista as pv

# %%
# Create Sphere Glyphs
# ~~~~~~~~~~~~~~~~~~~~

x = np.arange(-10, 10, 1, dtype=float)
y = np.arange(-10, 10, 1, dtype=float)
x, y = np.meshgrid(x, y)
r = np.sqrt(x**2 + y**2)
z = (np.sin(r) + 1) / 2

# Create and structured surface
grid = pv.StructuredGrid(x, y, z)
grid.point_data['size'] = z.ravel()

# generate glyphs with varying size
sphere = pv.Sphere(theta_resolution=16, phi_resolution=16)
spheres = grid.glyph(scale='size', geom=sphere, orient=False)

spheres.plot(show_scalar_bar=False)

# %%
# Create the Movie
# ~~~~~~~~~~~~~~~~

# Create a plotter object and set the scalars to the Z height
pl = pv.Plotter(notebook=False)
pl.add_mesh(
    spheres,
    show_edges=False,
    show_scalar_bar=False,
    clim=[0, 1],
    cmap='bwr',
)

# Open a gif
pl.open_gif('glyph_wave.gif')

# Update Z and write a frame for each updated mesh
nframe = 20
for phase in np.linspace(0, 2 * np.pi, nframe + 1)[:nframe]:
    z = (np.sin(r + phase) + 1) / 2

    # regenerate spheres
    grid = pv.StructuredGrid(x, y, z)
    grid.point_data['size'] = z.ravel()
    new_spheres = grid.glyph(scale='size', geom=sphere, orient=False)

    spheres.copy_from(new_spheres)

    # Write a frame. This triggers a render.
    pl.write_frame()

# Close and finalize the gif
pl.close()
# %%
# .. tags:: plot
