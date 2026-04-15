"""
.. _orbit_example:

Orbiting
~~~~~~~~

Orbit around a scene.

.. note::
   The quality of the movie will be better when using
   ``pl.open_movie('orbit.mp4')`` instead of
   ``pl.open_gif('orbit.gif')``

For orbiting to work you first have to show the scene and leave the plotter open
with ``.show(auto_close=False)``.  You may also have to set
``pv.Plotter(off_screen=True)``

.. note::
   Use ``lighting=False`` to reduce the size of the color space to avoid
   "jittery" GIFs when showing the scalar bar.

"""

# sphinx_gallery_thumbnail_number = 2
from __future__ import annotations

import pyvista as pv
from pyvista import examples

mesh = examples.download_st_helens().warp_by_scalar()

# %%
# Orbit around the Mt. St Helens dataset using
# :func:`~pyvista.Plotter.generate_orbital_path`.

pl = pv.Plotter()
pl.add_mesh(mesh, lighting=False)
pl.camera.zoom(1.5)
pl.show(auto_close=False)
path = pl.generate_orbital_path(n_points=36, shift=mesh.length)
pl.open_gif('orbit.gif')
pl.orbit_on_path(path, write_frames=True)
pl.close()


# %%

pl = pv.Plotter()
pl.add_mesh(mesh, lighting=False)
pl.show_grid()
pl.show(auto_close=False)
viewup = [0.5, 0.5, 1]
path = pl.generate_orbital_path(factor=2.0, shift=10000, viewup=viewup, n_points=36)
pl.open_gif('orbit.gif')
pl.orbit_on_path(path, write_frames=True, viewup=[0, 0, 1], step=0.05)
pl.close()


# %%

mesh = examples.download_dragon()
viewup = [0, 1, 0]

# %%
pl = pv.Plotter()
pl.add_mesh(mesh)
pl.show(auto_close=False)
path = pl.generate_orbital_path(factor=2.0, n_points=36, viewup=viewup, shift=0.2)
pl.open_gif('orbit.gif')
pl.orbit_on_path(path, write_frames=True, viewup=viewup, step=0.05)
pl.close()
# %%
# .. tags:: plot
