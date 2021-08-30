"""
.. _orbiting_example:

Orbiting
~~~~~~~~

Orbit around a scene.

.. note::
   The quality of the movie will be better when using
   ``p.open_movie('orbit.mp4')`` instead of
   ``p.open_gif('orbit.gif')``

For orbiting to work you first have to show the scene and leave the plotter open
with ``.show(auto_close=False)``.
"""

# sphinx_gallery_thumbnail_number = 2
import pyvista as pv
from pyvista import examples

mesh = examples.download_st_helens().warp_by_scalar()

###############################################################################

p = pv.Plotter()
p.add_mesh(mesh)
p.show(auto_close=False)
path = p.generate_orbital_path(n_points=36, shift=mesh.length)
p.open_gif("orbit.gif")
p.orbit_on_path(path, write_frames=True)
p.close()

###############################################################################

p = pv.Plotter()
p.add_mesh(mesh)
p.show_grid()
p.show(auto_close=False)
viewup = [0.5, 0.5, 1]
path = p.generate_orbital_path(factor=2.0, shift=10000, viewup=viewup, n_points=36)
p.open_gif("orbit.gif")
p.orbit_on_path(path, write_frames=True, viewup=[0, 0, 1], step=0.05)
p.close()


###############################################################################

mesh = examples.download_dragon()
viewup = [0, 1, 0]

###############################################################################
p = pv.Plotter()
p.add_mesh(mesh)
p.show(auto_close=False)
path = p.generate_orbital_path(factor=2.0, n_points=36, viewup=viewup, shift=0.2)
p.open_gif("orbit.gif")
p.orbit_on_path(path, write_frames=True, viewup=viewup, step=0.05)
p.close()
